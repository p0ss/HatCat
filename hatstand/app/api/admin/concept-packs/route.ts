// GET /api/admin/concept-packs — list concept packs from HatCatDev/concept_packs.

import { hatcatdevPath, listDirs } from "@/lib/server/hatcatdev";
import { fail, ok } from "@/lib/server/api-helpers";
import {
  CONCEPT_PACKS_DIR,
  readPackSummary,
} from "@/lib/server/concept-packs";
import type { ConceptPackSummary, Page } from "@/types";

export async function GET() {
  try {
    const root = hatcatdevPath(CONCEPT_PACKS_DIR);
    const names = await listDirs(root);
    const summaries = (
      await Promise.all(names.map((n) => readPackSummary(n)))
    ).filter((s): s is ConceptPackSummary => s !== null);

    summaries.sort((a, b) => {
      if (a.updated_at && b.updated_at) {
        return b.updated_at.localeCompare(a.updated_at);
      }
      return a.name.localeCompare(b.name);
    });

    const page: Page<ConceptPackSummary> = {
      items: summaries,
      total: summaries.length,
    };
    return ok(page, "filesystem:concept_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("concept_packs_list_failed", message);
  }
}
