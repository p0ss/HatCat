// GET /api/admin/lens-packs/[id]/calibration — full calibration view.
// Returns summary (over-firer counts, per-layer), the cycle progression, and
// a paginated/sortable/filterable list of per-(concept, layer) entries.

import { ok, notFound, fail, badRequest } from "@/lib/server/api-helpers";
import { pathExists } from "@/lib/server/hatcatdev";
import {
  listCalibrationEntries,
  loadCalibration,
  loadCalibrationCycles,
  packDir,
  summariseOverFirers,
} from "@/lib/server/lens-packs";
import type {
  CalibrationCycle,
  CalibrationOverFirerSummary,
  ConceptCalibration,
  Page,
} from "@/types";

type SortField =
  | "cross_fire_rate"
  | "noise_fire_rate"
  | "self_mean"
  | "concept";

const SORT_FIELDS = new Set<SortField>([
  "cross_fire_rate",
  "noise_fire_rate",
  "self_mean",
  "concept",
]);

export type CalibrationResponse = {
  pack_id: string;
  mode?: string;
  has_noise_track: boolean;
  summary: CalibrationOverFirerSummary;
  cycles: CalibrationCycle[];
  entries: Page<ConceptCalibration>;
};

export async function GET(
  req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  try {
    const dir = packDir(id);
    if (!(await pathExists(dir))) {
      return notFound(`Lens pack '${id}' not found`);
    }

    const cal = await loadCalibration(id);
    if (!cal) {
      return notFound(
        `Lens pack '${id}' has no calibration.json — has the calibration cycle run?`,
      );
    }

    const url = new URL(req.url);
    const overFirersOnly =
      (url.searchParams.get("over_firers_only") ?? "").toLowerCase() === "1" ||
      (url.searchParams.get("over_firers_only") ?? "").toLowerCase() === "true";
    const layerParam = url.searchParams.get("layer");
    const sortField = (url.searchParams.get("sort") ??
      "cross_fire_rate") as SortField;
    const sortDir = (url.searchParams.get("dir") ?? "desc").toLowerCase();
    const cursorParam = url.searchParams.get("cursor");
    const limitParam = url.searchParams.get("limit");

    if (!SORT_FIELDS.has(sortField)) {
      return badRequest(
        `Unknown sort field '${sortField}'. Allowed: ${[...SORT_FIELDS].join(", ")}`,
      );
    }

    let layer: number | undefined;
    if (layerParam !== null) {
      const parsed = Number(layerParam);
      if (!Number.isInteger(parsed)) {
        return badRequest("layer must be an integer");
      }
      layer = parsed;
    }

    let limit = 50;
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
        return badRequest("cursor must be a non-negative integer");
      }
      offset = Math.floor(parsed);
    }

    const allEntries = listCalibrationEntries(cal);
    const summary = summariseOverFirers(allEntries);

    const filtered = allEntries.filter((e) => {
      if (typeof layer === "number" && e.layer !== layer) return false;
      if (overFirersOnly) {
        const overCross = e.cross_fire_rate > summary.threshold;
        const overNoise =
          typeof e.noise_fire_rate === "number" &&
          e.noise_fire_rate > summary.threshold;
        if (!overCross && !overNoise) return false;
      }
      return true;
    });

    filtered.sort((a, b) => {
      let av: number | string;
      let bv: number | string;
      switch (sortField) {
        case "cross_fire_rate":
          av = a.cross_fire_rate;
          bv = b.cross_fire_rate;
          break;
        case "noise_fire_rate":
          av = a.noise_fire_rate ?? -1;
          bv = b.noise_fire_rate ?? -1;
          break;
        case "self_mean":
          av = a.self_mean;
          bv = b.self_mean;
          break;
        case "concept":
          av = `${a.concept}_${a.layer}`;
          bv = `${b.concept}_${b.layer}`;
          break;
      }
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });

    const slice = filtered.slice(offset, offset + limit);
    const cycles = await loadCalibrationCycles(id);

    const data: CalibrationResponse = {
      pack_id: id,
      mode: cal.mode,
      has_noise_track: typeof cal.noise_calibration_samples === "number",
      summary,
      cycles,
      entries: {
        items: slice,
        total: filtered.length,
        next_cursor:
          offset + slice.length < filtered.length
            ? String(offset + slice.length)
            : undefined,
      },
    };

    return ok(data, "filesystem:lens_packs/calibration");
  } catch (err) {
    return fail(
      "calibration_failed",
      err instanceof Error ? err.message : String(err),
      500,
    );
  }
}
