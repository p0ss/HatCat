// GET /api/admin/lens-packs/[id] — detail with aggregate metrics.
// Parses version_manifest.json (~1MB) once to compute avg test_f1 per layer.

import path from "node:path";
import { pathExists, statMtimeIso } from "@/lib/server/hatcatdev";
import { ok, notFound, fail } from "@/lib/server/api-helpers";
import {
  deriveCalibrationStatus,
  deriveStatus,
  distinctConceptCount,
  listCalibrationEntries,
  loadCalibration,
  loadPackInfo,
  loadRegistry,
  loadVersionManifest,
  packDir,
  summariseOverFirers,
  type LensVersionManifest,
} from "@/lib/server/lens-packs";
import type { LensPack, LensPackAggregateMetrics } from "@/types";

function computeAvgF1PerLayer(
  lenses: NonNullable<LensVersionManifest["lenses"]>,
): Record<string, number> {
  const sums: Record<string, { sum: number; n: number }> = {};
  for (const lens of Object.values(lenses)) {
    const layer = lens.default_layer ?? lens.layer;
    const f1 = lens.metrics?.f1;
    if (typeof layer !== "number" || typeof f1 !== "number") continue;
    const key = String(layer);
    if (!sums[key]) sums[key] = { sum: 0, n: 0 };
    sums[key].sum += f1;
    sums[key].n += 1;
  }
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(sums)) {
    out[k] = v.n === 0 ? 0 : v.sum / v.n;
  }
  return out;
}

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  try {
    const dir = packDir(id);
    if (!(await pathExists(dir))) {
      return notFound(`Lens pack '${id}' not found`);
    }

    const [registry, packInfo, calibration, manifest] = await Promise.all([
      loadRegistry(),
      loadPackInfo(id),
      loadCalibration(id),
      loadVersionManifest(id),
    ]);

    if (!packInfo && !registry?.packs?.[id]) {
      return notFound(`Lens pack '${id}' not found`);
    }

    const entry = registry?.packs?.[id];
    const hasCalibration = calibration !== null;
    const lenses = manifest?.lenses ?? {};
    const conceptCount = Object.keys(lenses).length;

    const calibrationEntries = listCalibrationEntries(calibration);
    const overSummary = summariseOverFirers(calibrationEntries);

    const aggregate_metrics: LensPackAggregateMetrics = {
      avg_test_f1_per_layer: computeAvgF1PerLayer(lenses),
      calibration_summary: calibration
        ? {
            entries_calibrated:
              calibration.total_concepts_calibrated ??
              calibrationEntries.length,
            concepts_calibrated: distinctConceptCount(calibrationEntries),
            concepts_total: conceptCount,
            over_firers: overSummary.over_firers,
            noise_over_firers: overSummary.noise_over_firers,
          }
        : undefined,
    };

    const updatedAt =
      manifest?.updated ??
      entry?.synced_at ??
      (await statMtimeIso(path.join(dir, "version_manifest.json"))) ??
      (await statMtimeIso(dir));

    const pack: LensPack = {
      id,
      substrate: packInfo?.model ?? "unknown",
      concept_pack: packInfo?.source_pack ?? "unknown",
      version: packInfo?.pack_version ?? entry?.version ?? "0.0.0",
      status: deriveStatus(packInfo, hasCalibration),
      calibration_status: deriveCalibrationStatus(
        packInfo,
        calibration,
        conceptCount,
      ),
      aggregate_metrics,
      registry_path: dir,
      created_at:
        entry?.created_at ?? manifest?.created ?? packInfo?.trained_at ?? "",
      updated_at:
        updatedAt ?? entry?.created_at ?? packInfo?.trained_at ?? "",
      based_on: packInfo?.based_on ?? entry?.based_on ?? undefined,
    };

    return ok(pack, "filesystem:lens_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("lens_pack_detail_failed", message);
  }
}
