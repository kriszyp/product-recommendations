/**
 * Wires up Harper's sourcedFrom caching for the Product table.
 * When ORIGIN_PRODUCT_API_URL is set, any miss on tables.Product.get(id)
 * automatically fetches from the origin API and stores the result.
 * Expiration is controlled by PRODUCT_CACHE_TTL_SECONDS (default: 24 h).
 *
 * The origin API shape is modelled after Salesforce Commerce Cloud:
 *   GET {ORIGIN_PRODUCT_API_URL}/products/{id}
 */

import { Resource, tables } from 'harper';

const ORIGIN_API_URL = process.env.ORIGIN_PRODUCT_API_URL ?? '';
const ORIGIN_API_KEY = process.env.ORIGIN_PRODUCT_API_KEY ?? '';

// ── Normalisation ─────────────────────────────────────────────────────────────

function tokenize(text: string): string[] {
	if (!text) return [];
	return text
		.toLowerCase()
		.replace(/[^\w\s]/g, ' ')
		.split(/\s+/)
		.filter((t) => t.length >= 3 && t.length <= 30);
}

export function normalizeOriginProduct(
	raw: Record<string, unknown>,
	productId: string,
) {
	const name = (raw.name as string) ?? (raw.pageTitle as string) ?? productId;
	const description =
		(raw.longDescription as string) ??
		(raw.shortDescription as string) ??
		(raw.description as string) ??
		'';
	const category =
		(raw.primaryCategoryId as string) ?? (raw.c_category as string) ?? '';
	const variants = raw.variants as Array<{ price?: number }> | undefined;
	const priceRanges = raw.priceRanges as Array<{ minPrice?: number }> | undefined;
	const price =
		(raw.price as number) ??
		variants?.[0]?.price ??
		priceRanges?.[0]?.minPrice ??
		null;
	const sku = (raw.id as string) ?? (raw.masterProductId as string) ?? productId;
	const imageGroups = raw.imageGroups as
		| Array<{ images: Array<{ link: string }> }>
		| undefined;
	const imageUrl =
		imageGroups?.[0]?.images?.[0]?.link ?? (raw.thumbnail as string) ?? '';

	const textContent = [
		...new Set(tokenize([name, description, category, sku].filter(Boolean).join(' '))),
	].join(' ');

	return {
		id: productId,
		name,
		description,
		category,
		price,
		sku,
		imageUrl,
		metadata: JSON.stringify(raw),
		fetchedAt: Date.now(),
		textContent,
	};
}

// ── Source resource (not exported → no HTTP endpoint) ─────────────────────────

class OriginProductAPI extends Resource {
	async get() {
		const productId = String(this.getId() ?? '');
		const url = `${ORIGIN_API_URL.replace(/\/$/, '')}/products/${encodeURIComponent(productId)}`;
		const headers: Record<string, string> = { Accept: 'application/json' };
		if (ORIGIN_API_KEY) headers['Authorization'] = `Bearer ${ORIGIN_API_KEY}`;

		const res = await fetch(url, { headers });
		if (!res.ok) {
			throw new Error(
				`Origin API returned ${res.status} for product ${productId}`,
			);
		}
		return normalizeOriginProduct(
			(await res.json()) as Record<string, unknown>,
			productId,
		);
	}
}

// Register origin as the cache source when a URL is configured
if (ORIGIN_API_URL) {
	(tables as any).Product.sourcedFrom(OriginProductAPI);
}
