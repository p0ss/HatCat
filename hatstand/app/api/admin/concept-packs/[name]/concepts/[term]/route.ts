// GET /api/admin/concept-packs/[name]/concepts/[term] — single concept detail.

import { pathExists } from "@/lib/server/hatcatdev";
import { ok, notFound, fail } from "@/lib/server/api-helpers";
import { loadConceptDetail, packDir } from "@/lib/server/concept-packs";

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ name: string; term: string }> },
) {
  const { name: rawName, term: rawTerm } = await ctx.params;
  const name = decodeURIComponent(rawName);
  const term = decodeURIComponent(rawTerm);
  try {
    const dir = packDir(name);
    if (!(await pathExists(dir))) {
      return notFound(`Concept pack '${name}' not found`);
    }
    const concept = await loadConceptDetail(dir, name, term);
    if (!concept) {
      return notFound(`Concept '${term}' not found in pack '${name}'`);
    }
    return ok(concept, "filesystem:concept_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("concept_detail_failed", message);
  }
}
