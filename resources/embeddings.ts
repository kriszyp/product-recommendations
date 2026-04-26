/**
 * Embedding generation utility — not a Harper Resource, no HTTP endpoint.
 *
 * Providers:
 *   EMBEDDING_PROVIDER=openai  → OpenAI text-embedding-3-small (1536 dims)
 *   EMBEDDING_PROVIDER=ollama  → Ollama nomic-embed-text (768 dims)
 *   EMBEDDING_PROVIDER=""      → disabled, returns null (Jaccard fallback remains active)
 *
 * All errors return null so the calling code degrades gracefully to text similarity.
 */

const PROVIDER = (process.env.EMBEDDING_PROVIDER ?? '').toLowerCase();
const OPENAI_MODEL = process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small';
const OLLAMA_HOST = (process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434').replace(/\/$/, '');
const OLLAMA_MODEL = process.env.OLLAMA_EMBEDDING_MODEL ?? 'nomic-embed-text';

export const EMBEDDING_VERSION = PROVIDER ? `${PROVIDER}-v1` : '';

export async function generateEmbedding(text: string): Promise<number[] | null> {
	if (!PROVIDER || !text) return null;
	try {
		if (PROVIDER === 'openai') return await _openaiEmbed(text);
		if (PROVIDER === 'ollama') return await _ollamaEmbed(text);
	} catch {
		// transient failure — caller falls back to Jaccard
	}
	return null;
}

async function _openaiEmbed(text: string): Promise<number[] | null> {
	const apiKey = process.env.OPENAI_API_KEY;
	if (!apiKey) return null;

	const res = await fetch('https://api.openai.com/v1/embeddings', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${apiKey}`,
		},
		body: JSON.stringify({ model: OPENAI_MODEL, input: text, encoding_format: 'float' }),
	});
	if (!res.ok) return null;
	const json = (await res.json()) as { data: Array<{ embedding: number[] }> };
	return json.data?.[0]?.embedding ?? null;
}

async function _ollamaEmbed(text: string): Promise<number[] | null> {
	const res = await fetch(`${OLLAMA_HOST}/api/embed`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ model: OLLAMA_MODEL, input: text }),
	});
	if (!res.ok) return null;
	const json = (await res.json()) as { embeddings?: number[][] };
	return json.embeddings?.[0] ?? null;
}
