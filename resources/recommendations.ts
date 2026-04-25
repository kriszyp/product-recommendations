import { Resource, tables, getContext } from 'harper';
import { normalizeOriginProduct } from './origin-cache.ts';

// ── Configuration ─────────────────────────────────────────────────────────────

const MAX_SESSION_HISTORY = Number(process.env.MAX_SESSION_HISTORY ?? 20);
const MAX_RECOMMENDATIONS = Number(process.env.MAX_RECOMMENDATIONS ?? 10);
const ASSOC_WINDOW = Number(process.env.ASSOC_WINDOW ?? 10);
const TEXT_SIM_THRESHOLD = Number(process.env.TEXT_SIM_THRESHOLD ?? 0.05);
const ASSOC_BOOTSTRAP_THRESHOLD = Number(process.env.ASSOC_BOOTSTRAP_THRESHOLD ?? 3);

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

async function incrementAssoc(from: string, to: string): Promise<void> {
	const id = `${from}>${to}`;
	try {
		const rec = await (tables as any).ProductAssociation.update(id);
		rec.addTo('weight', 1);
		rec.lastSeen = Date.now();
	} catch {
		await (tables as any).ProductAssociation
			.put(id, { id, productId: from, associatedProductId: to, weight: 1, lastSeen: Date.now() })
			.catch(() => {});
	}
}

async function updateAssociations(productId: string, priorIds: string[]): Promise<void> {
	await Promise.all(
		priorIds.map((otherId) =>
			Promise.all([incrementAssoc(productId, otherId), incrementAssoc(otherId, productId)]),
		),
	);
}

// ── Recommendation engine ─────────────────────────────────────────────────────

async function buildRecommendations(
	productId: string,
	currentProduct: Record<string, unknown> | null,
): Promise<unknown[]> {
	const scores = new Map<string, number>();

	// 1. Association-based scores (primary learned signal)
	const assocResults: Array<Record<string, unknown>> = [];
	for await (const assoc of (tables as any).ProductAssociation.search({
		conditions: [{ attribute: 'productId', comparator: 'eq', value: productId }],
		limit: 100,
	})) {
		assocResults.push(assoc as Record<string, unknown>);
	}
	assocResults.sort((a, b) => ((b.weight as number) ?? 0) - ((a.weight as number) ?? 0));
	for (const assoc of assocResults) {
		const target = assoc.associatedProductId as string;
		if (target && target !== productId) {
			scores.set(target, (scores.get(target) ?? 0) + (assoc.weight as number) * 2);
		}
	}

	// 2. Text similarity — bootstrapping when associations are sparse
	const strongAssocCount = [...scores.values()].filter((s) => s >= 2).length;
	if (strongAssocCount < ASSOC_BOOTSTRAP_THRESHOLD && currentProduct?.textContent) {
		const currentTokens = tokenize(currentProduct.textContent as string);
		for await (const other of (tables as any).Product.search({ limit: 500 })) {
			const otherId = (other as any).id as string;
			if (!otherId || otherId === productId) continue;
			const sim = jaccardSim(
				currentTokens,
				tokenize(((other as any).textContent as string) ?? (other as any).name ?? ''),
			);
			if (sim >= TEXT_SIM_THRESHOLD) {
				scores.set(otherId, (scores.get(otherId) ?? 0) + sim * 5);
			}
		}
	}

	scores.delete(productId);

	const topIds = [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_RECOMMENDATIONS);

	return Promise.all(
		topIds.map(async ([id, score]) => {
			const p = (await (tables as any).Product.get(id).catch(() => null)) as Record<
				string,
				unknown
			> | null;
			return {
				id,
				name: p?.name ?? id,
				description: p?.description != null ? (p.description as string).slice(0, 200) : '',
				price: p?.price ?? null,
				category: p?.category ?? '',
				imageUrl: p?.imageUrl ?? '',
				score: Math.round((score as number) * 100) / 100,
			};
		}),
	);
}

// ── Core handler (shared by get and post) ─────────────────────────────────────

async function handle(productId: string, productName: string) {
	const session = (getContext() as any)?.session as Record<string, unknown> | undefined;
	const history: string[] = (session?.productHistory as string[]) ?? [];

	const product = await ensureProduct(productId, productName);

	if (!history.includes(productId)) {
		await updateAssociations(productId, history.slice(0, ASSOC_WINDOW));
	}

	const newHistory = [productId, ...history.filter((id) => id !== productId)].slice(
		0,
		MAX_SESSION_HISTORY,
	);
	if (session) session.productHistory = newHistory;

	const recs = await buildRecommendations(productId, product);

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
