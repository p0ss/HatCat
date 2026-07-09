// GET /api/admin/search — universal search across all indexed resources.
//
// Query params:
//   q                — full-text query
//   type             — resource_type filter (model, concept, lens_pack, etc.)
//   filter[<facet>]  — facet filter, repeatable; values must match exactly
//   cursor           — opaque pagination token (offset)
//   limit            — default 25, capped at 200

import { fail, ok } from "@/lib/server/api-helpers";
import { getIndex, invalidateIndex } from "@/lib/server/search/build";
import type { FacetKey, ResourceType, SearchResponse } from "@/types";

const KNOWN_RESOURCE_TYPES = new Set<ResourceType>([
  "model",
  "concept_pack",
  "concept",
  "lens_pack",
  "lens",
  "simplex",
  "meld",
  "run",
  "doc",
]);

function parseFilter(searchParams: URLSearchParams): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const [key, value] of searchParams.entries()) {
    const m = /^filter\[([A-Za-z0-9_\-]+)\]$/.exec(key);
    if (!m) continue;
    const facet = m[1];
    (out[facet] ??= []).push(value);
  }
  return out;
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const q = url.searchParams.get("q") ?? undefined;
    const typeParam = url.searchParams.get("type") ?? undefined;
    const cursor = url.searchParams.get("cursor");
    const limitRaw = url.searchParams.get("limit");

    let type: ResourceType | undefined;
    if (typeParam) {
      if (!KNOWN_RESOURCE_TYPES.has(typeParam as ResourceType)) {
        return fail("bad_type", `Unknown resource_type '${typeParam}'`, 400);
      }
      type = typeParam as ResourceType;
    }

    const offset = cursor ? Math.max(0, Number(cursor) || 0) : 0;
    const limit = limitRaw ? Math.max(1, Math.min(200, Number(limitRaw) || 25)) : 25;
    const filter = parseFilter(url.searchParams) as Partial<
      Record<FacetKey, string | string[]>
    >;

    const store = await getIndex();
    const result: SearchResponse = store.search({ q, type, filter, offset, limit });
    return ok(result, `index:${store.size()}`);
  } catch (err) {
    return fail(
      "search_failed",
      err instanceof Error ? err.message : String(err),
      500,
    );
  }
}

export async function POST(req: Request) {
  // POST /api/admin/search/reindex isn't a real subroute since Next.js
  // app router scopes by file. We instead accept POST here as a "reindex now"
  // signal — bodyless, safe to call.
  if (new URL(req.url).searchParams.get("reindex") === "1") {
    invalidateIndex();
    const store = await getIndex();
    return ok({ rebuilt: true, total: store.size() }, "reindex");
  }
  return fail("method_not_allowed", "POST requires ?reindex=1", 405);
}
