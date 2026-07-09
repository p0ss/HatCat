// GET /api/admin/concept-packs/[name] — single pack metadata.

import { pathExists } from "@/lib/server/hatcatdev";
import { ok, notFound, fail } from "@/lib/server/api-helpers";
import { packDir, readPackSummary } from "@/lib/server/concept-packs";

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ name: string }> },
) {
  const { name: rawName } = await ctx.params;
  const name = decodeURIComponent(rawName);
  try {
    if (!(await pathExists(packDir(name)))) {
      return notFound(`Concept pack '${name}' not found`);
    }
    const summary = await readPackSummary(name);
    if (!summary) {
      return notFound(`Concept pack '${name}' has no pack.json`);
    }
    return ok(summary, "filesystem:concept_packs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("concept_pack_detail_failed", message);
  }
}
