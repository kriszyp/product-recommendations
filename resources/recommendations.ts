import { Resource, tables, getContext } from 'harper';
import { normalizeOriginProduct } from './origin-cache.ts';

// ── Configuration ─────────────────────────────────────────────────────────────

const MAX_SESSION_HISTORY = Number(process.env.MAX_SESSION_HISTORY ?? 20);
const MAX_RECOMMENDATIONS = Number(process.env.MAX_RECOMMENDATIONS ?? 10);
const ASSOC_WINDOW = Number(process.env.ASSOC_WINDOW ?? 10);
const ASSOC_WEIGHT = Number(process.env.ASSOC_WEIGHT ?? 10);
const TEXT_SIM_THRESHOLD = Number(process.env.TEXT_SIM_THRESHOLD ?? 0.05);
const ASSOC_BOOTSTRAP_THRESHOLD = Number(process.env.ASSOC_BOOTSTRAP_THRESHOLD ?? 3);

// Phase 1: temporal decay
const DECAY_HALF_LIFE_MS = Number(process.env.DECAY_HALF_LIFE_MS ?? 2_592_000_000); // 30 days

// Phase 3: second-order graph traversal
const SECOND_ORDER_THRESHOLD = Number(process.env.SECOND_ORDER_THRESHOLD ?? MAX_RECOMMENDATIONS);
const SECOND_ORDER_DISCOUNT = Number(process.env.SECOND_ORDER_DISCOUNT ?? 0.3);
const SECOND_ORDER_HOPS = Number(process.env.SECOND_ORDER_HOPS ?? 5);

// Phase 4: diversity re-ranking
const DIVERSITY_OVERSAMPLE = Number(process.env.DIVERSITY_OVERSAMPLE ?? 3);
const CATEGORY_PENALTY = Number(process.env.CATEGORY_PENALTY ?? 0.5);
const MAX_PER_CATEGORY = Number(process.env.MAX_PER_CATEGORY ?? 3);

// Phase 5: vector embeddings
const SEMANTIC_WEIGHT = Number(process.env.SEMANTIC_WEIGHT ?? 5);
const SEMANTIC_CANDIDATES = Number(process.env.SEMANTIC_CANDIDATES ?? 50);
const VECTOR_SIM_THRESHOLD = Number(process.env.VECTOR_SIM_THRESHOLD ?? 0.6);

// Phase 6: UCB-style exploration — EXPLORE_WEIGHT=0 disables entirely (no behaviour change)
const EXPLORE_WEIGHT = Number(process.env.EXPLORE_WEIGHT ?? 0);
const EXPLORE_FORCE_CANDIDATES = Number(process.env.EXPLORE_FORCE_CANDIDATES ?? MAX_RECOMMENDATIONS);

// Phase 7: session context blending — SESSION_BLEND_WEIGHT=0 disables entirely
const SESSION_BLEND_WEIGHT = Number(process.env.SESSION_BLEND_WEIGHT ?? 0.5);
const SESSION_BLEND_WINDOW = Number(process.env.SESSION_BLEND_WINDOW ?? 3);

// ── Text similarity helpers ───────────────────────────────────────────────────

function tokenize(text: string): Set<string> {
	if (!text) return new Set();
	return new Set(
		text
			.toLowerCase()
			.replace(/[^\w\s]/g, ' ')
			.split(/\s+/)
			.filter((t) => t.length >= 3 && t.length <= 30),
	);
}

function jaccardSim(a: Set<string>, b: Set<string>): number {
	if (!a.size || !b.size) return 0;
	let inter = 0;
	for (const t of a) if (b.has(t)) inter++;
	return inter / (a.size + b.size - inter);
}

// Phase 1: exponential decay — older associations contribute less
function decayedWeight(weight: number, lastSeen: number): number {
	const age = Math.max(0, Date.now() - (lastSeen ?? 0));
	return weight * Math.pow(2, -age / DECAY_HALF_LIFE_MS);
}

// ── Product cache ─────────────────────────────────────────────────────────────

async function ensureProduct(
	productId: string,
	fallbackName: string,
): Promise<Record<string, unknown> | null> {
	const product = await (tables as any).Product.get(productId).catch(() => null);
	if (product) return product;

	if (fallbackName) {
		const minimal = normalizeOriginProduct({ name: fallbackName, id: productId }, productId);
		await (tables as any).Product.put(productId, minimal).catch(() => {});
		return minimal;
	}

	return null;
}

// ── Association learning ──────────────────────────────────────────────────────

// Phase 1: delta is float (recency-weighted increment)
async function incrementAssocBy(from: string, to: string, delta: number): Promise<void> {
	const id = `${from}>${to}`;
	try {
		const rec = await (tables as any).ProductAssociation.update(id);
		rec.addTo('weight', delta);
		rec.lastSeen = Date.now();
	} catch {
		await (tables as any).ProductAssociation
			.put(id, { id, productId: from, associatedProductId: to, weight: delta, lastSeen: Date.now() })
			.catch(() => {});
	}
	// Phase 2: maintain per-product aggregate for PMI normalization
	await touchProductStats(from, delta);
}

// Phase 1: weight by inverse position — most recent session item counts most
async function updateAssociations(productId: string, priorIds: string[]): Promise<void> {
	await Promise.all(
		priorIds.map((otherId, idx) => {
			const delta = 1 / (idx + 1); // 1.0, 0.5, 0.33, 0.25 …
			return Promise.all([
				incrementAssocBy(productId, otherId, delta),
				incrementAssocBy(otherId, productId, delta),
			]);
		}),
	);
}

// ── Phase 2: product popularity stats for PMI normalization ───────────────────

async function touchProductStats(productId: string, delta: number): Promise<void> {
	try {
		const s = await (tables as any).ProductStats.update(productId);
		s.addTo('totalCoOccurrences', delta);
		s.lastUpdated = Date.now();
	} catch {
		await (tables as any).ProductStats
			.put(productId, { id: productId, totalCoOccurrences: delta, lastUpdated: Date.now() })
			.catch(() => {});
	}
}

// ── Recommendation engine ─────────────────────────────────────────────────────

async function buildRecommendations(
	productId: string,
	currentProduct: Record<string, unknown> | null,
	sessionHistory: string[],
): Promise<unknown[]> {
	const scores = new Map<string, number>();
	const sessionHistorySet = new Set(sessionHistory);

	// ── Signal 1: association scoring (Phase 1 decay + Phase 2 PMI) ───────────

	const assocResults: Array<Record<string, unknown>> = [];
	for await (const assoc of (tables as any).ProductAssociation.search({
		conditions: [{ attribute: 'productId', comparator: 'eq', value: productId }],
		limit: 100,
	})) {
		assocResults.push(assoc as Record<string, unknown>);
	}

	// Phase 2: batch-fetch stats for PMI normalization
	const [currentStats, ...candidateStats] = await Promise.all([
		(tables as any).ProductStats.get(productId).catch(() => null),
		...assocResults.map((a) =>
			(tables as any).ProductStats.get(a.associatedProductId as string).catch(() => null),
		),
	]);
	const currentTotal = Math.max(1, (currentStats?.totalCoOccurrences as number) ?? 1);

	assocResults.sort((a, b) => ((b.weight as number) ?? 0) - ((a.weight as number) ?? 0));
	for (let i = 0; i < assocResults.length; i++) {
		const assoc = assocResults[i];
		const target = assoc.associatedProductId as string;
		if (!target || target === productId) continue;
		const targetTotal = Math.max(1, (candidateStats[i]?.totalCoOccurrences as number) ?? 1);
		// Phase 1 decay × Phase 2 PMI normalization
		const dw = decayedWeight(assoc.weight as number, assoc.lastSeen as number);
		const pmiScore = dw / Math.sqrt(currentTotal * targetTotal);
		scores.set(target, (scores.get(target) ?? 0) + pmiScore * ASSOC_WEIGHT);
	}

	// ── Signal 1b: session context blending (Phase 7) ────────────────────────
	// Traverse the association graph from the user's recent session products and
	// blend their scores in. Makes recommendations reflect the browsing journey
	// (e.g. "shoes + socks + gels" session surfaces running-themed candidates),
	// not just the single product being viewed right now.

	if (SESSION_BLEND_WEIGHT > 0 && sessionHistory.length > 0) {
		const blendSources = sessionHistory.slice(0, SESSION_BLEND_WINDOW);

		const blendAssocLists = await Promise.all(
			blendSources.map(async (sourceId) => {
				const results: Array<Record<string, unknown>> = [];
				for await (const assoc of (tables as any).ProductAssociation.search({
					conditions: [{ attribute: 'productId', comparator: 'eq', value: sourceId }],
					limit: 50,
				})) {
					results.push(assoc as Record<string, unknown>);
				}
				return results;
			}),
		);

		// Batch-fetch stats for all sources and their candidates in one round-trip
		const blendStatsIdSet = new Set<string>(blendSources);
		for (const assocs of blendAssocLists) {
			for (const a of assocs) blendStatsIdSet.add(a.associatedProductId as string);
		}
		const blendStatsIds = [...blendStatsIdSet];
		const blendStatsArr = await Promise.all(
			blendStatsIds.map((id) => (tables as any).ProductStats.get(id).catch(() => null)),
		);
		const blendStatsMap = new Map(blendStatsIds.map((id, i) => [id, blendStatsArr[i]]));

		for (let k = 0; k < blendSources.length; k++) {
			const blendWeight = SESSION_BLEND_WEIGHT / (k + 1);
			const sourceTotal = Math.max(
				1,
				(blendStatsMap.get(blendSources[k])?.totalCoOccurrences as number) ?? 1,
			);

			for (const assoc of blendAssocLists[k]) {
				const target = assoc.associatedProductId as string;
				if (!target || target === productId) continue;
				const targetTotal = Math.max(
					1,
					(blendStatsMap.get(target)?.totalCoOccurrences as number) ?? 1,
				);
				const dw = decayedWeight(assoc.weight as number, assoc.lastSeen as number);
				const pmiScore = dw / Math.sqrt(sourceTotal * targetTotal);
				scores.set(target, (scores.get(target) ?? 0) + pmiScore * ASSOC_WEIGHT * blendWeight);
			}
		}
	}

	// ── Signal 2: semantic similarity (Phase 5 vector or Jaccard fallback) ────

	const strongAssocCount = [...scores.values()].filter((s) => s >= 1).length;
	const embedding = currentProduct?.textEmbedding as number[] | undefined;

	if (embedding && Array.isArray(embedding) && embedding.length > 0) {
		// Phase 5: O(log n) HNSW vector search
		for await (const similar of (tables as any).Product.search({
			select: ['id', '$distance'],
			sort: { attribute: 'textEmbedding', target: embedding },
			limit: SEMANTIC_CANDIDATES,
		})) {
			const otherId = (similar as any).id as string;
			if (!otherId || otherId === productId) continue;
			const distance = (similar as any).$distance as number; // cosine distance 0..2
			const sim = 1 - distance / 2; // convert to 0..1 similarity
			if (sim >= VECTOR_SIM_THRESHOLD) {
				scores.set(otherId, (scores.get(otherId) ?? 0) + sim * SEMANTIC_WEIGHT);
			}
		}
	} else if (strongAssocCount < ASSOC_BOOTSTRAP_THRESHOLD && currentProduct?.textContent) {
		// Jaccard fallback: O(n) scan, only used during cold-start
		const currentTokens = tokenize(currentProduct.textContent as string);
		for await (const other of (tables as any).Product.search({ limit: 500 })) {
			const otherId = (other as any).id as string;
			if (!otherId || otherId === productId) continue;
			const sim = jaccardSim(
				currentTokens,
				tokenize(((other as any).textContent as string) ?? (other as any).name ?? ''),
			);
			if (sim >= TEXT_SIM_THRESHOLD) {
				scores.set(otherId, (scores.get(otherId) ?? 0) + sim * SEMANTIC_WEIGHT);
			}
		}
	}

	// ── Signal 3: second-order graph traversal (Phase 3) ─────────────────────

	if (scores.size < SECOND_ORDER_THRESHOLD) {
		const topFirstHop = [...scores.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, SECOND_ORDER_HOPS)
			.map(([id, score]) => ({ id, score }));

		await Promise.all(
			topFirstHop.map(async ({ id: hopId, score: hopScore }) => {
				for await (const assoc of (tables as any).ProductAssociation.search({
					conditions: [{ attribute: 'productId', comparator: 'eq', value: hopId }],
					limit: 20,
				})) {
					const target = (assoc as any).associatedProductId as string;
					// Only add second-hop candidates not already reached directly
					if (!target || target === productId || scores.has(target)) continue;
					const dw = decayedWeight(
						(assoc as any).weight as number,
						(assoc as any).lastSeen as number,
					);
					scores.set(target, hopScore * dw * SECOND_ORDER_DISCOUNT);
				}
			}),
		);
	}

	scores.delete(productId);

	// Remove products already viewed this session — no value recommending what the user just saw
	for (const viewedId of sessionHistorySet) {
		scores.delete(viewedId);
	}

	// ── Signal 4: UCB exploration bonus (Phase 6) ────────────────────────────
	// Injects products never recommended before so the association graph can't
	// self-reinforce into a closed loop of already-popular products.

	let rawScores: Map<string, number> | undefined;
	let exploreStatsMap: Map<string, Record<string, unknown> | null> | undefined;

	if (EXPLORE_WEIGHT > 0) {
		// Forced-exploration candidates: products with no association/similarity signal
		if (EXPLORE_FORCE_CANDIDATES > 0) {
			const forcedIds = new Set<string>();
			for await (const p of (tables as any).Product.search({ limit: EXPLORE_FORCE_CANDIDATES * 3 })) {
				const pid = (p as any).id as string;
				if (pid && pid !== productId && !scores.has(pid) && !sessionHistorySet.has(pid)) {
					forcedIds.add(pid);
					if (forcedIds.size >= EXPLORE_FORCE_CANDIDATES) break;
				}
			}
			for (const pid of forcedIds) scores.set(pid, 0);
		}

		// Batch-fetch ProductStats for all candidates.
		// Seed from the already-fetched candidateStats to avoid redundant reads.
		exploreStatsMap = new Map<string, Record<string, unknown> | null>();
		assocResults.forEach((a, i) =>
			exploreStatsMap!.set(a.associatedProductId as string, candidateStats[i]),
		);
		const missingIds = [...scores.keys()].filter((id) => !exploreStatsMap!.has(id));
		const missingStats = await Promise.all(
			missingIds.map((id) => (tables as any).ProductStats.get(id).catch(() => null)),
		);
		missingIds.forEach((id, i) => exploreStatsMap!.set(id, missingStats[i]));

		// Save raw exploitation scores, then apply UCB bonus
		rawScores = new Map(scores);
		for (const [id, raw] of scores) {
			const impressions = Math.max(
				0,
				(exploreStatsMap.get(id)?.recommendationImpressions as number) ?? 0,
			);
			scores.set(id, raw + EXPLORE_WEIGHT / Math.sqrt(1 + impressions));
		}
	}

	// ── Phase 4: diversity re-ranking ─────────────────────────────────────────

	const oversampleCount = MAX_RECOMMENDATIONS * DIVERSITY_OVERSAMPLE;
	const allEntries = [...scores.entries()].sort((a, b) => b[1] - a[1]);
	const maxScore = allEntries[0]?.[1] ?? 1;
	const minScore = allEntries[allEntries.length - 1]?.[1] ?? 0;
	const scoreRange = maxScore - minScore || 1;

	const candidates = allEntries.slice(0, oversampleCount).map(([id, raw]) => ({
		id,
		normalizedScore: ((raw - minScore) / scoreRange) * 100,
	}));

	// Fetch product details for the candidate pool in one parallel batch
	const candidateProducts = await Promise.all(
		candidates.map((c) =>
			(tables as any).Product.get(c.id).catch(() => null) as Promise<Record<
				string,
				unknown
			> | null>,
		),
	);
	const productMap = new Map(
		candidates.map((c, i) => [c.id, candidateProducts[i]]),
	);

	// Greedy MMR-style category re-ranking
	const categoryCounts = new Map<string, number>();
	const selected: Array<{ id: string; normalizedScore: number }> = [];
	const used = new Set<string>();

	while (selected.length < MAX_RECOMMENDATIONS && selected.length < candidates.length) {
		let bestIdx = -1;
		let bestScore = -Infinity;

		for (let i = 0; i < candidates.length; i++) {
			if (used.has(candidates[i].id)) continue;
			const cat = (productMap.get(candidates[i].id)?.category as string) ?? '__unknown__';
			const catCount = categoryCounts.get(cat) ?? 0;
			if (catCount >= MAX_PER_CATEGORY) continue;
			const penalized = candidates[i].normalizedScore * Math.pow(CATEGORY_PENALTY, catCount);
			if (penalized > bestScore) {
				bestScore = penalized;
				bestIdx = i;
			}
		}

		if (bestIdx === -1) break;

		selected.push({ id: candidates[bestIdx].id, normalizedScore: Math.round(bestScore * 100) / 100 });
		used.add(candidates[bestIdx].id);
		const cat = (productMap.get(candidates[bestIdx].id)?.category as string) ?? '__unknown__';
		categoryCounts.set(cat, (categoryCounts.get(cat) ?? 0) + 1);
	}

	// Mark items where the exploration bonus exceeded the exploitation score
	const explorationBoosted = new Set<string>(
		EXPLORE_WEIGHT > 0 && rawScores && exploreStatsMap
			? selected.map((c) => c.id).filter((id) => {
					const raw = rawScores!.get(id) ?? 0;
					const impressions = Math.max(
						0,
						(exploreStatsMap!.get(id)?.recommendationImpressions as number) ?? 0,
					);
					return EXPLORE_WEIGHT / Math.sqrt(1 + impressions) > raw;
				})
			: [],
	);

	return selected.map(({ id, normalizedScore }) => {
		const p = productMap.get(id);
		return {
			id,
			name: p?.name ?? id,
			description: p?.description != null ? (p.description as string).slice(0, 200) : '',
			price: p?.price ?? null,
			category: p?.category ?? '',
			imageUrl: p?.imageUrl ?? '',
			score: normalizedScore,
			explored: explorationBoosted.has(id),
		};
	});
}

// ── Core handler (shared by get and post) ─────────────────────────────────────

async function handle(productId: string, productName: string) {
	const session = (getContext() as any)?.session as Record<string, unknown> | undefined;
	const history: string[] = (session?.productHistory as string[]) ?? [];

	const product = await ensureProduct(productId, productName);

	if (!history.includes(productId)) {
		await updateAssociations(productId, history.slice(0, ASSOC_WINDOW));
	}

	const priorViews = history.filter((id) => id !== productId);
	const newHistory = [productId, ...priorViews].slice(0, MAX_SESSION_HISTORY);
	if (session) session.productHistory = newHistory;

	const recs = await buildRecommendations(productId, product, priorViews);

	// Fire-and-forget: increment recommendationImpressions for each returned product.
	// Never blocks the response — mirrors the embedding generation pattern in origin-cache.ts.
	Promise.all(
		(recs as Array<{ id: string }>).map((rec) =>
			(async () => {
				try {
					const s = await (tables as any).ProductStats.update(rec.id);
					s.addTo('recommendationImpressions', 1);
				} catch {
					await (tables as any).ProductStats
						.put(rec.id, {
							id: rec.id,
							totalCoOccurrences: 0,
							recommendationImpressions: 1,
							lastUpdated: Date.now(),
						})
						.catch(() => {});
				}
			})(),
		),
	).catch(() => {});

	return {
		productId,
		product: product
			? {
					id: product.id,
					name: product.name,
					description: product.description,
					category: product.category,
					price: product.price,
					imageUrl: product.imageUrl,
				}
			: { id: productId },
		recommendations: recs,
		sessionHistory: newHistory,
	};
}

// ── Resource ──────────────────────────────────────────────────────────────────

export class recommendations extends Resource {
	// GET /recommendations/{productId}
	static async get(target: any) {
		const productId = String(target?.id ?? target ?? '');
		if (!productId) {
			return new Response(
				JSON.stringify({ error: 'Product ID required: GET /recommendations/{productId}' }),
				{ status: 400, headers: { 'Content-Type': 'application/json' } },
			);
		}
		return handle(productId, '');
	}

	// POST /recommendations/  body: { productId, productName? }
	static async post(_target: unknown, body: Record<string, string>) {
		const { productId, productName = '' } = body ?? {};
		if (!productId) {
			return new Response(
				JSON.stringify({ error: 'productId is required' }),
				{ status: 400, headers: { 'Content-Type': 'application/json' } },
			);
		}
		return handle(productId, productName);
	}
}
