// GET /api/admin/lens-packs/[id]/simplexes — simplex inventory for a pack.

import { pathExists } from "@/lib/server/hatcatdev";
import { ok, notFound, fail } from "@/lib/server/api-helpers";
import {
  getSimplexInventoryInfo,
  listPackSimplexes,
  packDir,
  type SimplexInventoryInfo,
} from "@/lib/server/lens-packs";
import type { Simplex } from "@/types";

export type SimplexInventoryResponse = {
  pack_id: string;
  info: SimplexInventoryInfo;
  simplexes: Simplex[];
};

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  try {
    if (!(await pathExists(packDir(id)))) {
      return notFound(`Lens pack '${id}' not found`);
    }
    const [info, simplexes] = await Promise.all([
      getSimplexInventoryInfo(id),
      listPackSimplexes(id),
    ]);
    const data: SimplexInventoryResponse = {
      pack_id: id,
      info,
      simplexes,
    };
    return ok(data, "filesystem:lens_packs/simplexes");
  } catch (err) {
    return fail(
      "simplexes_failed",
      err instanceof Error ? err.message : String(err),
      500,
    );
  }
}
