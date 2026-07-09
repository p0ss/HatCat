// GET /api/admin/lens-packs — list lens packs.
// List view reads only registry + per-pack pack_info.json + calibration.json
// presence to keep this fast; per-concept manifest parsing is on the detail.

import path from "node:path";
import { pathExists, statMtimeIso } from "@/lib/server/hatcatdev";
import { fail, ok } from "@/lib/server/api-helpers";
import {
  deriveCalibrationStatus,
  deriveStatus,
  loadPackInfo,
  loadRegistry,
  packDir,
} from "@/lib/server/lens-packs";
import type { LensPack, Page } from "@/types";

export async function GET() {
  try {
    const registry = await loadRegistry();
    const ids = Object.keys(registry.packs ?? {});

    const items: LensPack[] = await Promise.all(
      ids.map(async (id) => {
        const dir = packDir(id);
        const packInfo = await loadPackInfo(id);
        const hasCalibration = await pathExists(
          path.join(dir, "calibration.json"),
        );
        const updatedAt =
          (await statMtimeIso(path.join(dir, "pack_info.json"))) ??
          (await statMtimeIso(dir));
        const entry = registry.packs?.[id];
        return {
          id,
          substrate: packInfo?.model ?? "unknown",
          concept_pack: packInfo?.source_pack ?? "unknown",
          version: packInfo?.pack_version ?? entry?.version ?? "0.0.0",
          status: deriveStatus(packInfo, hasCalibration),
          calibration_status: deriveCalibrationStatus(
            packInfo,
            hasCalibration ? { total_concepts_calibrated: 1 } : null,
          ),
          aggregate_metrics: { avg_test_f1_per_layer: {} },
          registry_path: dir,
          created_at: entry?.created_at ?? packInfo?.trained_at ?? "",
          updated_at:
            entry?.synced_at ??
            updatedAt ??
            entry?.created_at ??
            packInfo?.trained_at ??
            "",
          based_on: packInfo?.based_on ?? entry?.based_on ?? undefined,
        };
      }),
    );

    items.sort((a, b) => {
      if (a.updated_at && b.updated_at) {
        return b.updated_at.localeCompare(a.updated_at);
      }
      return a.id.localeCompare(b.id);
    });

    const page: Page<LensPack> = { items, total: items.length };
    return ok(page, "filesystem:lens_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("lens_packs_list_failed", message);
  }
}
