import { Resource, tables } from 'harperdb';
import { normalizeOriginProduct } from './origin-cache.ts';

// ── Configuration ─────────────────────────────────────────────────────────────

const MAX_SESSION_HISTORY = Number(process.env.MAX_SESSION_HISTORY ?? 20);
const MAX_RECOMMENDATIONS = Number(process.env.MAX_RECOMMENDATIONS ?? 10);
// How many prior session products to associate with each new product view
const ASSOC_WINDOW = Number(process.env.ASSOC_WINDOW ?? 10);
// Minimum Jaccard similarity threshold for text-based bootstrapping
const TEXT_SIM_THRESHOLD = Number(process.env.TEXT_SIM_THRESHOLD ?? 0.05);
// Number of strong associations before text similarity is skipped
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

// ── Main resource ─────────────────────────────────────────────────────────────

export class recommendations extends Resource {
	// Public endpoint — no authentication required
	allowRead() {
		return true;
	}
	allowCreate() {
		return true;
	}

	// GET /recommendations/{productId}
	async get() {
		const productId = String(this.getId() ?? '');
		if (!productId) {
			return new Response(
				JSON.stringify({ error: 'Product ID required: GET /recommendations/{productId}' }),
				{ status: 400, headers: { 'Content-Type': 'application/json' } },
			);
		}
		return this._handle(productId, '');
	}

	// POST /recommendations/  body: { productId, productName? }
	async post(_target: unknown, body: Record<string, string>) {
		const { productId, productName = '' } = body ?? {};
		if (!productId) {
			return new Response(
				JSON.stringify({ error: 'productId is required' }),
				{ status: 400, headers: { 'Content-Type': 'application/json' } },
			);
		}
		return this._handle(productId, productName);
	}

	private async _handle(productId: string, productName: string) {
		// Retrieve session — Harper cookie-based sessions store arbitrary data
		const session = (this.getContext() as any)?.session as
			| Record<string, unknown>
			| undefined;
		const history: string[] = (session?.productHistory as string[]) ?? [];

		// Fetch / cache product details (sourcedFrom handles origin API automatically)
		const product = await this._ensureProduct(productId, productName);

		// Build associations: new product ↔ recent session products
		if (!history.includes(productId)) {
			await this._updateAssociations(productId, history.slice(0, ASSOC_WINDOW));
		}

		// Prepend current product to session history and truncate
		const newHistory = [
			productId,
			...history.filter((id) => id !== productId),
		].slice(0, MAX_SESSION_HISTORY);
		if (session) session.productHistory = newHistory;

		const recs = await this._buildRecommendations(productId, product);

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

	// ── Product cache ────────────────────────────────────────────────────────
	// Harper auto-fetches from origin via sourcedFrom (origin-cache.ts).
	// This method ensures a minimal record exists even when origin is unavailable.

	private async _ensureProduct(
		productId: string,
		fallbackName: string,
	): Promise<Record<string, unknown> | null> {
		const product = await (tables as any).Product.get(productId).catch(
			() => null,
		);
		if (product) return product;

		// Origin unavailable — bootstrap a minimal record from the provided name
		if (fallbackName) {
			const minimal = normalizeOriginProduct(
				{ name: fallbackName, id: productId },
				productId,
			);
			await (tables as any).Product.put(productId, minimal).catch(() => {});
			return minimal;
		}

		return null;
	}

	// ── Association learning ─────────────────────────────────────────────────

	private async _updateAssociations(
		productId: string,
		priorIds: string[],
	): Promise<void> {
		await Promise.all(
			priorIds.map((otherId) =>
				Promise.all([
					this._incrementAssoc(productId, otherId),
					this._incrementAssoc(otherId, productId),
				]),
			),
		);
	}

	private async _incrementAssoc(from: string, to: string): Promise<void> {
		const id = `${from}>${to}`;
		try {
			// Atomic increment on existing record
			const rec = await (tables as any).ProductAssociation.update(id);
			rec.addTo('weight', 1);
			rec.lastSeen = Date.now();
		} catch {
			// Record doesn't exist yet — create it
			await (tables as any).ProductAssociation
				.put(id, {
					id,
					productId: from,
					associatedProductId: to,
					weight: 1,
					lastSeen: Date.now(),
				})
				.catch(() => {});
		}
	}

	// ── Recommendation engine ────────────────────────────────────────────────

	private async _buildRecommendations(
		productId: string,
		currentProduct: Record<string, unknown> | null,
	): Promise<unknown[]> {
		const scores = new Map<string, number>();

		// 1. Association-based scores (primary learned signal, grows over time)
		const assocResults: Array<Record<string, unknown>> = [];
		for await (const assoc of (tables as any).ProductAssociation.search({
			conditions: [
				{ attribute: 'productId', comparator: 'eq', value: productId },
			],
			limit: 100,
		})) {
			assocResults.push(assoc as Record<string, unknown>);
		}
		assocResults.sort(
			(a, b) => ((b.weight as number) ?? 0) - ((a.weight as number) ?? 0),
		);
		for (const assoc of assocResults) {
			const target = assoc.associatedProductId as string;
			if (target && target !== productId) {
				scores.set(
					target,
					(scores.get(target) ?? 0) + (assoc.weight as number) * 2,
				);
			}
		}

		// 2. Text similarity — bootstrapping signal when associations are sparse
		const strongAssocCount = [...scores.values()].filter((s) => s >= 2).length;
		if (
			strongAssocCount < ASSOC_BOOTSTRAP_THRESHOLD &&
			currentProduct?.textContent
		) {
			const currentTokens = tokenize(currentProduct.textContent as string);
			for await (const other of (tables as any).Product.search({ limit: 500 })) {
				const otherId = (other as any).id as string;
				if (!otherId || otherId === productId) continue;
				const sim = jaccardSim(
					currentTokens,
					tokenize(
						((other as any).textContent as string) ??
							(other as any).name ??
							'',
					),
				);
				if (sim >= TEXT_SIM_THRESHOLD) {
					scores.set(otherId, (scores.get(otherId) ?? 0) + sim * 5);
				}
			}
		}

		scores.delete(productId);

		const topIds = [...scores.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, MAX_RECOMMENDATIONS);

		return Promise.all(
			topIds.map(async ([id, score]) => {
				const p = (await (tables as any).Product.get(id).catch(
					() => null,
				)) as Record<string, unknown> | null;
				return {
					id,
					name: p?.name ?? id,
					description:
						p?.description != null
							? (p.description as string).slice(0, 200)
							: '',
					price: p?.price ?? null,
					category: p?.category ?? '',
					imageUrl: p?.imageUrl ?? '',
					score: Math.round((score as number) * 100) / 100,
				};
			}),
		);
	}
}
