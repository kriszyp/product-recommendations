# Product Recommendations Engine

A Harper application that delivers real-time, low-latency product recommendations. The engine combines multiple learned and content-based signals that improve continuously as traffic grows, with no offline training pipeline or external ML infrastructure required.

---

## Why Harper Is a Good Fit

### Low-latency at the edge

Deployed to a Harper Fabric cluster, the product graph and product cache are replicated across every node globally. A user in Tokyo and a user in London both read from their nearest node — no round-trip to a central database. Recommendation lookups are served from local in-process storage.

### Associations update in real time across the cluster

Harper's replication propagates association weight increments across the cluster automatically. A browsing pattern that appears on nodes in Frankfurt immediately strengthens the same association edges on nodes in Singapore. The recommendation model improves globally without any centralised aggregation step or batch retraining.

### Composition with Harper's page and API caching

This component is designed to sit alongside Harper's built-in caching components. A typical e-commerce setup might look like:

```
Browser
  │
  ▼
Harper Node (nearest)
  ├─ Static assets        (Harper static component — served from edge)
  ├─ Product detail page  (Harper HTTP cache — full page or fragment cache)
  ├─ Product API          (Harper cache component — cached origin API responses)
  └─ /recommendations/    (this component — live, session-aware, no cache)
```

The recommendation endpoint is intentionally not cached at the HTTP layer — each request is session-specific and must read the current association graph. However, the underlying `Product` and `ProductAssociation` table lookups are served from Harper's in-process storage, so individual record reads are already as fast as a cache hit.

For higher-traffic deployments, recommendations for anonymous (cookieless) users can be cached per product ID at the page layer, since those requests carry no session context. Personalised responses bypass the page cache automatically.

### No external ML infrastructure required

The recommendation model lives entirely inside Harper tables. There is no separate vector database, no offline training pipeline, no model serving infrastructure, and no scheduled jobs. The association graph and popularity statistics are updated inline with every request using Harper's atomic increment operations. Deploying the recommendation engine is just deploying a Harper component.

---

## Recommendation Techniques

The engine stacks seven complementary techniques. Each one addresses a different failure mode of simple co-occurrence counting, and they compose cleanly because they all operate on the same Harper tables.

### 1. Session co-occurrence graph (primary signal)

Every time a user views a product, the engine records a co-occurrence between that product and everything else the user looked at earlier in the same session. These associations are stored as weighted directed edges in the `ProductAssociation` table (`A→B` and `B→A` stored separately for fast single-key queries). Over time this builds a product graph reflecting real browsing behaviour — users who look at running shoes also look at compression socks, users researching laptops also view monitors and keyboards.

**Recency weighting within a session:** Not all co-occurrences are equally informative. A product viewed moments ago is a stronger signal than one viewed ten clicks back. Each edge increment uses an inverse-position weight: `delta = 1 / (position + 1)`, so position 0 (most recent) contributes `1.0`, position 1 contributes `0.5`, position 2 contributes `0.33`, and so on. This is applied atomically via Harper's `table.update(id).addTo('weight', delta)`, so concurrent requests from different nodes never produce corrupt weights.

### 2. Temporal decay

Raw co-occurrence counts accumulate forever, meaning a browsing pattern from two years ago carries the same weight as one from yesterday. The engine applies **exponential decay** at query time using the `lastSeen` timestamp that is already stored on every association record:

```
effectiveWeight = storedWeight × 2^( −age / halfLife )
```

With a 30-day half-life (configurable via `DECAY_HALF_LIFE_MS`), an association that has not been reinforced in 30 days contributes half as much as a fresh one; after 90 days it contributes one eighth. Seasonal products naturally fade between seasons and re-emerge when browsing patterns return. No batch job or scheduled cleanup is needed — the decay is a pure computation at read time.

### 3. Popularity normalization (PMI-style)

A naive co-occurrence graph over-recommends globally popular products. A bestselling item that appears in thousands of sessions will accumulate high edge weights to almost every other product, even ones it has no real affinity with. This is the recommendation equivalent of a stopword problem.

The fix is a lightweight approximation of **Pointwise Mutual Information** normalized by product popularity:

```
score(A→B) = decayedWeight(A→B) / √( totalCoOccurrences(A) × totalCoOccurrences(B) )
```

`totalCoOccurrences` is maintained per product in a `ProductStats` table, incremented atomically alongside every edge write. Dividing by the geometric mean of both products' marginal frequencies means the score is high only when A and B appear together *more often than their individual popularities would predict* — the true definition of association.

### 4. Second-order graph traversal (friends-of-friends)

Direct associations only exist between products that have literally appeared in the same session. New products have no direct edges. Products in niche categories may have very few sessions. In both cases, the first-hop graph alone produces too few candidates to fill the recommendation list.

The engine expands the top-5 first-hop neighbours one level further, accumulating indirect candidates at a discounted weight:

```
indirectScore(A→C) = score(A→B) × decayedWeight(B→C) × discount
```

`SECOND_ORDER_DISCOUNT` defaults to 0.3. The `scores.has(target)` guard ensures second-hop scores never overwrite stronger direct scores. Worst-case cost is `5 × 20 = 100` additional Harper reads per request — all served from local replicated storage, sub-millisecond each.

### 5. Semantic similarity (cold-start bootstrapping)

The association graph is empty on day one and sparse for new products regardless of how much general traffic the system receives. The engine supplements association scores with a content-based similarity signal to ensure useful recommendations are returned even before any learning has occurred.

**Jaccard similarity (no dependencies):** The default. Each product's `textContent` field stores a deduplicated token set built from name, description, category, and SKU. Jaccard similarity — intersection size divided by union size — between the current product's tokens and every cached product is computed in a single pass. Candidates above `TEXT_SIM_THRESHOLD` (default `0.05`) contribute `similarity × SEMANTIC_WEIGHT` to the score map. This requires no external services and works immediately after deployment.

**HNSW vector similarity (optional, much more accurate):** When `EMBEDDING_PROVIDER` is set, the `textContent` string is sent to an embedding API (OpenAI `text-embedding-3-small` or Ollama `nomic-embed-text`) and the resulting vector is stored in a `textEmbedding: [Float]` field that carries an `@indexed(type: "HNSW", distance: "cosine")` directive. Harper maintains the HNSW index incrementally. At query time, a single approximate nearest-neighbour search replaces the O(n) Jaccard scan:

```
vectorScore(B) = 1 − cosineDistance(embedding(A), embedding(B)) / 2
```

Embeddings are generated **lazily on first encounter** and persisted via a fire-and-forget patch — they never block the response. Existing products gain embeddings as their cache entries cycle through normal expiration. The Jaccard fallback remains active for products whose embeddings have not yet been generated.

The key advantage of vector embeddings over Jaccard: "running shoes" and "jogging sneakers" have zero token overlap but high cosine similarity. Synonyms, paraphrases, and cross-language product names all resolve correctly.

### 6. Diversity re-ranking

Score-ranked top-K lists frequently cluster in one category. A user viewing a laptop might receive ten monitors and zero keyboards, mice, or bags — technically high-scoring but not a useful list.

After building the scored candidate pool, the engine **over-fetches** `MAX_RECOMMENDATIONS × DIVERSITY_OVERSAMPLE` (default 3×) candidates and applies a greedy Maximal Marginal Relevance-style re-rank:

- **Soft penalty:** each additional recommendation from the same `category` already in the final list has its score multiplied by `CATEGORY_PENALTY^count` (default `0.5` — halves per repeat)
- **Hard cap:** `MAX_PER_CATEGORY` slots per category (default 3) — no category can monopolise the list

Products without a `category` field share an `__unknown__` bucket. The re-ranker is O(candidates × targetCount) with category data fetched in the same `Promise.all` batch as the final product detail reads, so it adds no extra round-trips.

### 7. UCB-style exploration (anti-feedback-loop)

The association graph has a self-reinforcing property: highly-associated products get recommended → users view them → their edge weights grow → they get recommended even more. Products that never enter the graph never accumulate associations and stay invisible forever, regardless of their actual relevance.

The engine applies a **Upper Confidence Bound (UCB)-style exploration bonus** before normalization to counter this:

```
finalScore = rawScore + EXPLORE_WEIGHT × ( 1 / √(1 + impressions) )
```

`impressions` is the number of times a product has appeared in a recommendation response, tracked atomically in `ProductStats.recommendationImpressions`. The bonus is large when a product is unseen (`impressions = 0` → bonus = 1.0) and shrinks as it accumulates exposure (`impressions = 99` → bonus ≈ 0.1). Products that have earned strong associations are not suppressed — their exploitation score outweighs the bonus of less-seen alternatives.

**Forced cold-start candidates:** Setting `EXPLORE_FORCE_CANDIDATES > 0` injects products with no prior association or semantic signal (raw score = 0) directly into the candidate pool before the diversity re-ranker. Without this, a product that has never been recommended can never earn impressions and is permanently invisible to the UCB formula. Forced candidates are given `score = 0` then lifted by the exploration bonus, so they can only appear if the bonus is large enough to survive the diversity re-rank.

**Zero overhead when disabled:** `EXPLORE_WEIGHT=0` (the default) skips the entire block — no extra reads, no allocation, identical behaviour to a deployment without the feature.

Each recommendation response includes an `explored` field per item, set to `true` when the exploration bonus exceeded the raw exploitation score. This lets callers render exploration slots differently (e.g. "You might also like" vs "Frequently bought together") and supports offline analysis of whether explored items convert.

### How the signals combine

All signals feed into a single `scores` map. The final score going into diversity re-ranking is:

| Signal | Formula | Scale |
|---|---|---|
| Association (PMI-normalized) | `decayedWeight(A→B) / √(totalA × totalB) × ASSOC_WEIGHT` | 0–10 |
| Second-order (indirect) | `firstHopScore × decayedWeight(B→C) × SECOND_ORDER_DISCOUNT` | 0–3 |
| Semantic (vector or Jaccard) | `similarity × SEMANTIC_WEIGHT` | 0–5 |
| Exploration bonus | `EXPLORE_WEIGHT / √(1 + impressions)` | 0–`EXPLORE_WEIGHT` |

Before the diversity re-ranker, scores are min-max normalized to a 0–100 range across the candidate pool. The `score` field in the API response reflects this normalized value.

---

## API

### `GET /recommendations/{productId}`

Returns recommendations for the given product. Automatically fetches and caches product details from the origin API if the product hasn't been seen before.

```sh
curl http://localhost:9926/recommendations/prod-abc-123
```

### `POST /recommendations/`

Same as GET but accepts a body, useful when the product name is known before the origin API has been called (bootstraps the text similarity signal immediately).

```sh
curl -X POST http://localhost:9926/recommendations/ \
  -H 'Content-Type: application/json' \
  -d '{ "productId": "prod-abc-123", "productName": "Trail Running Shoes" }'
```

### Response shape

```json
{
  "productId": "prod-abc-123",
  "product": {
    "id": "prod-abc-123",
    "name": "Trail Running Shoes",
    "description": "...",
    "category": "Footwear",
    "price": 129.99,
    "imageUrl": "..."
  },
  "recommendations": [
    {
      "id": "prod-def-456",
      "name": "Compression Running Socks",
      "description": "...",
      "price": 24.99,
      "category": "Accessories",
      "imageUrl": "...",
      "score": 84.2,
      "explored": false
    },
    {
      "id": "prod-new-789",
      "name": "Trail Running Vest",
      "description": "...",
      "price": 59.99,
      "category": "Apparel",
      "imageUrl": "...",
      "score": 31.4,
      "explored": true
    }
  ],
  "sessionHistory": ["prod-abc-123", "prod-xyz-789"]
}
```

`score` is normalized to 0–100 within the response's candidate pool. `explored: true` means the exploration bonus exceeded the raw exploitation score for that item — the engine is deliberately surfacing an under-exposed product. When `EXPLORE_WEIGHT=0` (the default), `explored` is always `false`.

---

## Schema

### `Product` table

Cached product details from the origin API. Expires after `PRODUCT_CACHE_TTL_SECONDS` (default 24 h); stale records are refreshed on next access via `sourcedFrom`.

| Field | Type | Notes |
|---|---|---|
| `id` | ID | Primary key |
| `name` | String | Indexed |
| `description` | String | |
| `price` | Float | |
| `category` | String | Indexed; used for diversity re-ranking |
| `sku` | String | |
| `imageUrl` | String | |
| `metadata` | String | Raw origin API response (JSON) |
| `fetchedAt` | Float | Unix ms timestamp of last fetch |
| `textContent` | String | Deduplicated token string for Jaccard similarity |
| `textEmbedding` | [Float] | HNSW-indexed vector; populated lazily when `EMBEDDING_PROVIDER` is set |
| `embeddingVersion` | String | Provider + model tag; used to detect stale vectors |

### `ProductAssociation` table

Learned co-occurrence edges. Both directions `A→B` and `B→A` are stored so queries only need to filter on `productId`.

| Field | Type | Notes |
|---|---|---|
| `id` | ID | `"{productId}>{associatedProductId}"` |
| `productId` | ID | Indexed |
| `associatedProductId` | ID | Indexed |
| `weight` | Float | Recency-weighted co-occurrence sum; incremented atomically |
| `lastSeen` | Float | Timestamp of last co-occurrence; used for temporal decay |

### `ProductStats` table

Per-product aggregate statistics for PMI normalization and exploration tracking. Access is always by primary key.

| Field | Type | Notes |
|---|---|---|
| `id` | ID | Mirrors `Product.id` |
| `totalCoOccurrences` | Float | Running sum of all outbound association increments |
| `lastUpdated` | Float | |
| `recommendationImpressions` | Float | Times this product appeared in a recommendation response; `null` treated as 0 |

---

## Configuration

All tuning parameters are set via environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `ORIGIN_PRODUCT_API_URL` | — | Base URL for origin product API (Salesforce Commerce Cloud compatible) |
| `ORIGIN_PRODUCT_API_KEY` | — | Bearer token for origin API |
| `PRODUCT_CACHE_TTL_SECONDS` | `86400` | Product cache lifetime in seconds |
| `MAX_SESSION_HISTORY` | `20` | Max product IDs tracked per session |
| `ASSOC_WINDOW` | `10` | How many prior session products to associate with each new view |
| `ASSOC_WEIGHT` | `10` | Score multiplier on PMI-normalized association score |
| `DECAY_HALF_LIFE_MS` | `2592000000` | Temporal decay half-life (default 30 days) |
| `SECOND_ORDER_THRESHOLD` | `10` | Candidate count below which second-hop expansion fires |
| `SECOND_ORDER_DISCOUNT` | `0.3` | Score discount for second-hop candidates |
| `SECOND_ORDER_HOPS` | `5` | Top-N first-hop neighbours to expand |
| `MAX_RECOMMENDATIONS` | `10` | Max recommendations returned per request |
| `ASSOC_BOOTSTRAP_THRESHOLD` | `3` | Strong-association count above which semantic similarity is skipped |
| `TEXT_SIM_THRESHOLD` | `0.05` | Minimum Jaccard similarity for text-based candidates |
| `DIVERSITY_OVERSAMPLE` | `3` | Candidate over-fetch multiplier before diversity re-rank |
| `CATEGORY_PENALTY` | `0.5` | Score multiplier per repeated category in output list |
| `MAX_PER_CATEGORY` | `3` | Hard cap on recommendations from the same category |
| `EXPLORE_WEIGHT` | `0` | UCB exploration bonus multiplier; 0 = disabled. Setting equal to `SEMANTIC_WEIGHT` (5) is a good starting point |
| `EXPLORE_FORCE_CANDIDATES` | `10` | Max products with no prior signal to inject as exploration candidates per request; 0 = only boost already-scored products |
| `EMBEDDING_PROVIDER` | — | `"openai"`, `"ollama"`, or unset (Jaccard fallback) |
| `OPENAI_API_KEY` | — | Required when `EMBEDDING_PROVIDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `VECTOR_SIM_THRESHOLD` | `0.6` | Minimum cosine similarity for vector candidates |
| `SEMANTIC_CANDIDATES` | `50` | Number of HNSW neighbours to retrieve |
| `SEMANTIC_WEIGHT` | `5` | Score multiplier on semantic similarity signal |

---

## Getting Started

### Install Harper

```sh
npm install -g harper
```

### Run locally

```sh
npm run dev
```

The recommendations endpoint will be available at `http://localhost:9926/recommendations/`.

### Configure the origin API (optional)

Copy `.env.example` to `.env` and set `ORIGIN_PRODUCT_API_URL`. Without it the engine works immediately — products are bootstrapped from the `productName` field in POST requests and association learning starts from the first session.

### Enable semantic embeddings (optional)

Set `EMBEDDING_PROVIDER=ollama` (local, free) or `EMBEDDING_PROVIDER=openai` (higher quality) in `.env`. Embeddings are generated lazily — existing recommendations continue working via Jaccard until each product's embedding has been populated.

### Deploy to Harper Fabric

Create a cluster at [https://fabric.harper.fast/](https://fabric.harper.fast/), add your cluster credentials to `.env`, then:

```sh
npm run deploy
```

The component deploys with rolling restarts and full replication across all nodes in your cluster.
