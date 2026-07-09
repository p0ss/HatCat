// GET /api/admin/lens-packs/[id]/lenses — paginated/filterable lens inventory.
//
// Query params:
//   q       — substring (case-insensitive) match on concept term
//   layer   — filter to a single model layer
//   sort    — concept | layer | f1 | training_samples
//   dir     — asc | desc
//   cursor  — offset
//   limit   — default 50, max 500

import { pathExists } from "@/lib/server/hatcatdev";
import { ok, notFound, fail, badRequest } from "@/lib/server/api-helpers";
import {
  listManifestLenses,
  packDir,
  type LensWithMeta,
} from "@/lib/server/lens-packs";
import type { Page } from "@/types";

const SORT_FIELDS = new Set([
  "concept",
  "layer",
  "f1",
  "training_samples",
] as const);
type SortField = "concept" | "layer" | "f1" | "training_samples";

export type LensesResponse = {
  pack_id: string;
  entries: Page<LensWithMeta>;
};

export async function GET(
  req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  try {
    if (!(await pathExists(packDir(id)))) {
      return notFound(`Lens pack '${id}' not found`);
    }

    const url = new URL(req.url);
    const q = url.searchParams.get("q")?.toLowerCase() ?? undefined;
    const layerParam = url.searchParams.get("layer");
    const sort = (url.searchParams.get("sort") ?? "f1") as SortField;
    const dir = (url.searchParams.get("dir") ?? "desc").toLowerCase();
    const cursorParam = url.searchParams.get("cursor");
    const limitParam = url.searchParams.get("limit");

    if (!SORT_FIELDS.has(sort)) {
      return badRequest(
        `Unknown sort field '${sort}'. Allowed: ${[...SORT_FIELDS].join(", ")}`,
      );
    }

    let layer: number | undefined;
    if (layerParam !== null) {
      const parsed = Number(layerParam);
      if (!Number.isInteger(parsed)) return badRequest("layer must be integer");
      layer = parsed;
    }

    let limit = 50;
    if (limitParam !== null) {
      const parsed = Number(limitParam);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return badRequest("limit must be positive");
      }
      limit = Math.min(500, Math.floor(parsed));
    }

    let offset = 0;
    if (cursorParam !== null) {
      const parsed = Number(cursorParam);
      if (!Number.isFinite(parsed) || parsed < 0) {
        return badRequest("cursor must be non-negative");
      }
      offset = Math.floor(parsed);
    }

    const all = await listManifestLenses(id);
    const filtered = all.filter((l) => {
      if (typeof layer === "number" && l.selected_layer !== layer) return false;
      if (q && !l.term.toLowerCase().includes(q)) return false;
      return true;
    });

    filtered.sort((a, b) => {
      let av: number | string;
      let bv: number | string;
      switch (sort) {
        case "concept":
          av = a.term;
          bv = b.term;
          break;
        case "layer":
          av = a.selected_layer;
          bv = b.selected_layer;
          break;
        case "f1":
          av = a.training_metrics.test_f1;
          bv = b.training_metrics.test_f1;
          break;
        case "training_samples":
          av = a.training_samples ?? -1;
          bv = b.training_samples ?? -1;
          break;
      }
      if (typeof av === "number" && typeof bv === "number") {
        return dir === "asc" ? av - bv : bv - av;
      }
      return dir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });

    const slice = filtered.slice(offset, offset + limit);
    const next = offset + slice.length;
    const data: LensesResponse = {
      pack_id: id,
      entries: {
        items: slice,
        total: filtered.length,
        next_cursor: next < filtered.length ? String(next) : undefined,
      },
    };
    return ok(data, "filesystem:lens_packs/lenses");
  } catch (err) {
    return fail(
      "lenses_failed",
      err instanceof Error ? err.message : String(err),
      500,
    );
  }
}
