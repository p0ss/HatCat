// GET /api/admin/concept-packs/[name]/concepts — paginated, filterable concept list.
//
// Supports query params:
//   layer  — number, filter to a single layer
//   q      — substring (case-insensitive) across term/lemmas/definition
//   cursor — opaque pagination token (offset)
//   limit  — default 100, capped at 500

import { pathExists } from "@/lib/server/hatcatdev";
import { ok, notFound, fail, badRequest } from "@/lib/server/api-helpers";
import { loadAllConcepts, packDir } from "@/lib/server/concept-packs";
import type { Concept, Page } from "@/types";

function matchesQuery(concept: Concept, q: string): boolean {
  const needle = q.toLowerCase();
  if (concept.term.toLowerCase().includes(needle)) return true;
  if (concept.definition?.toLowerCase().includes(needle)) return true;
  for (const lemma of concept.lemmas) {
    if (lemma.toLowerCase().includes(needle)) return true;
  }
  return false;
}

export async function GET(
  req: Request,
  ctx: { params: Promise<{ name: string }> },
) {
  const { name: rawName } = await ctx.params;
  const name = decodeURIComponent(rawName);
  try {
    const url = new URL(req.url);
    const layerParam = url.searchParams.get("layer");
    const q = url.searchParams.get("q") ?? undefined;
    const cursorParam = url.searchParams.get("cursor");
    const limitParam = url.searchParams.get("limit");

    let layer: number | undefined;
    if (layerParam !== null) {
      const parsed = Number(layerParam);
      if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
        return badRequest("layer must be an integer");
      }
      layer = parsed;
    }

    let limit = 100;
    if (limitParam !== null) {
      const parsed = Number(limitParam);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return badRequest("limit must be a positive number");
      }
      limit = Math.min(500, Math.floor(parsed));
    }

    let offset = 0;
    if (cursorParam !== null) {
      const parsed = Number(cursorParam);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return badRequest("cursor must be a non-negative number");
      }
      offset = Math.floor(parsed);
    }

    const dir = packDir(name);
    if (!(await pathExists(dir))) {
      return notFound(`Concept pack '${name}' not found`);
    }

    const all = await loadAllConcepts(dir, name);
    const filtered = all.filter((c) => {
      if (typeof layer === "number" && c.layer !== layer) return false;
      if (q && !matchesQuery(c, q)) return false;
      return true;
    });

    filtered.sort((a, b) => {
      if (a.layer !== b.layer) return a.layer - b.layer;
      return a.term.localeCompare(b.term);
    });

    const slice = filtered.slice(offset, offset + limit);
    const next = offset + slice.length;
    const page: Page<Concept> = {
      items: slice,
      total: filtered.length,
      next_cursor: next < filtered.length ? String(next) : undefined,
    };
    return ok(page, "filesystem:concept_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("concept_packs_concepts_failed", message);
  }
}
