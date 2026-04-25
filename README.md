# Product Recommendations Engine

A Harper application that delivers real-time, low-latency product recommendations by combining two complementary signals: **learned session associations** that improve continuously over time, and **text similarity bootstrapping** that produces useful results from day one.

## How It Works

### Dual-Signal Recommendation Engine

**1. Session-based association learning (primary signal)**

Every time a user views a product, the engine records a co-occurrence association between that product and everything else the user looked at earlier in the same session. These associations are stored as weighted edges in a `ProductAssociation` table — the more often two products appear together in sessions, the higher their shared weight. Over time this builds a product graph that reflects real browsing behaviour: users who buy running shoes also look at compression socks, users researching laptops also view monitors and keyboards.

**2. Text similarity bootstrapping (cold-start signal)**

Before enough session data has accumulated, the engine falls back to Jaccard similarity on tokenised product text (name, description, category, SKU). This ensures new products and new deployments produce meaningful recommendations immediately, without waiting for traffic to build up the association graph. Once the association graph has enough strong edges for a given product, the text scan is skipped entirely.

### Product Caching

Product details are fetched from a configurable external origin API (modelled after the Salesforce Commerce Cloud product API) and cached in Harper's `Product` table. Harper's `sourcedFrom` mechanism handles cache misses transparently — any `Product.get(id)` that isn't already cached triggers a background fetch from the origin API and stores the result. Cache entries expire after 24 hours (configurable) and are refreshed on next access. If the origin API is unavailable, stale cache is served rather than failing.

### Session Tracking

Each user's session is tracked using Harper's built-in cookie-based sessions (`getContext().session`). The session stores a rolling list of recently viewed product IDs. Associations are built incrementally from this list on every request, so the recommendation graph continuously improves without any batch processing or scheduled jobs.

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
      "score": 18.4
    }
  ],
  "sessionHistory": ["prod-abc-123", "prod-xyz-789"]
}
```

The `score` field reflects a combination of association weight and text similarity — higher means stronger signal.

---

## Why Harper Is a Good Fit

### Low-latency at the edge

Deployed to a Harper Fabric cluster, the product graph and product cache are replicated across every node globally. A user in Tokyo and a user in London both read from their nearest node — no round-trip to a central database. Recommendation lookups are served from local memory-mapped storage.

### Associations update in real time across the cluster

Harper's replication propagates association weight increments across the cluster automatically. A browsing pattern that appears on nodes in Frankfurt immediately strengthens the same association edges on nodes in Singapore. The recommendation model improves globally without any centralised aggregation step.

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

The product recommendation endpoint is intentionally **not cached at the HTTP layer** — each request is session-specific and must read the current association graph. However, the underlying `Product` table lookups that happen inside the endpoint are served from Harper's in-process storage (effectively in-memory), so the individual record reads are already as fast as a cache hit.

For higher-traffic deployments, the recommendations response for anonymous (cookieless) users could be cached at the page level per product ID, since those requests carry no session context. Personalised responses (users with session history) bypass the page cache automatically.

### No external ML infrastructure required

The recommendation model lives entirely inside Harper tables. There is no separate vector database, no offline training pipeline, no model serving infrastructure. The association graph is updated inline with every request. This makes the system operationally simple: deploying the recommendation engine is just deploying a Harper component.

### Extensibility with vector search

When richer semantic similarity is needed — for example, to recommend visually similar products or handle synonymous search terms — Harper's built-in HNSW vector index can be added to the `Product` table. Embeddings generated by OpenAI, Ollama, or any other provider can be stored alongside product records, and Harper's vector search replaces the Jaccard text scan with a single ANN query. The association learning layer works identically regardless of which similarity method is used underneath.

---

## Schema

### `Product` table

Cached product details from the origin API. Expires after `PRODUCT_CACHE_TTL_SECONDS` (default 24 h); stale records are refreshed on next access via `sourcedFrom`.

| Field | Type | Notes |
|---|---|---|
| `id` | ID | Product ID (primary key) |
| `name` | String | Indexed for lookup |
| `description` | String | |
| `price` | Float | |
| `category` | String | Indexed |
| `sku` | String | |
| `imageUrl` | String | |
| `metadata` | String | Raw origin API response (JSON) |
| `fetchedAt` | Float | Unix ms timestamp of last fetch |
| `textContent` | String | Tokenised text used for Jaccard similarity |

### `ProductAssociation` table

Learned co-occurrence edges. A record exists for every ordered pair `(A, B)` that has ever appeared in the same session — both `A→B` and `B→A` are stored, so queries only need to filter on `productId`.

| Field | Type | Notes |
|---|---|---|
| `id` | ID | `"{productId}>{associatedProductId}"` |
| `productId` | ID | Indexed |
| `associatedProductId` | ID | Indexed |
| `weight` | Float | Co-occurrence count; incremented atomically |
| `lastSeen` | Float | Unix ms timestamp of last co-occurrence |

---

## Configuration

All tuning parameters are set via environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `ORIGIN_PRODUCT_API_URL` | — | Base URL for origin product API (Salesforce Commerce Cloud compatible) |
| `ORIGIN_PRODUCT_API_KEY` | — | Bearer token for origin API (omit if not required) |
| `PRODUCT_CACHE_TTL_SECONDS` | `86400` | Product cache lifetime in seconds |
| `MAX_SESSION_HISTORY` | `20` | Max product IDs tracked per session |
| `ASSOC_WINDOW` | `10` | How many prior session products to associate with each new view |
| `MAX_RECOMMENDATIONS` | `10` | Max recommendations returned per request |
| `TEXT_SIM_THRESHOLD` | `0.05` | Minimum Jaccard similarity for text-based candidates |
| `ASSOC_BOOTSTRAP_THRESHOLD` | `3` | Strong-association count above which text similarity is skipped |

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

Copy `.env.example` to `.env` and set `ORIGIN_PRODUCT_API_URL`. Without it, the engine works immediately — products are bootstrapped from the `productName` field in POST requests and association learning starts from the first session.

### Deploy to Harper Fabric

Create a cluster at [https://fabric.harper.fast/](https://fabric.harper.fast/), add your cluster credentials to `.env`, then:

```sh
npm run deploy
```

The component will be deployed with rolling restarts and replication enabled across all nodes in your cluster.
