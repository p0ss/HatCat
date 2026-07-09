// GET /api/admin/registry — combined lens-pack + concept-pack registry summary.
// Detail / per-resource shapes live on their own routes.

import { hatcatdevPath, listDirs, tryReadJson } from "@/lib/server/hatcatdev";
import { ok, fail } from "@/lib/server/api-helpers";
import {
  loadPackInfo,
  loadRegistry,
  packDir as lensPackDir,
} from "@/lib/server/lens-packs";
import { CONCEPT_PACKS_DIR } from "@/lib/server/concept-packs";
import path from "node:path";
import type { LensPackRegistryEntry, Registry } from "@/types";

type ConceptPackJson = {
  pack_id?: string;
  version?: string;
};

export async function GET() {
  try {
    // ---------- Lens packs ----------
    const registry = await loadRegistry();
    const lensIds = Object.keys(registry.packs ?? {});
    const lens_packs: LensPackRegistryEntry[] = await Promise.all(
      lensIds.map(async (id) => {
        const dir = lensPackDir(id);
        const info = await loadPackInfo(id);
        const entry = registry.packs?.[id];
        return {
          id,
          path: dir,
          substrate: info?.model ?? "unknown",
          concept_pack: info?.source_pack ?? "unknown",
          version: info?.pack_version ?? entry?.version ?? "0.0.0",
        };
      }),
    );
    lens_packs.sort((a, b) => a.id.localeCompare(b.id));

    // ---------- Concept packs ----------
    const conceptDirs = await listDirs(hatcatdevPath(CONCEPT_PACKS_DIR));
    const concept_packs: Registry["concept_packs"] = [];
    for (const name of conceptDirs) {
      const dir = hatcatdevPath(CONCEPT_PACKS_DIR, name);
      const pack = await tryReadJson<ConceptPackJson>(
        path.join(dir, "pack.json"),
      );
      if (!pack) continue;
      concept_packs.push({
        name: pack.pack_id ?? name,
        path: dir,
        version: pack.version ?? "0.0.0",
      });
    }
    concept_packs.sort((a, b) => a.name.localeCompare(b.name));

    return ok<Registry>({ lens_packs, concept_packs }, "hatcatdev");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("registry_failed", message);
  }
}
