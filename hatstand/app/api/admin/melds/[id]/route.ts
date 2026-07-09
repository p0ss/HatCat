// GET /api/admin/melds/[id] — locate the meld file across pending/reference/applied
// and return the full Meld record.

import { fail, notFound, ok } from "@/lib/server/api-helpers";
import { findMeldById } from "@/lib/server/melds";

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  try {
    const { id: rawId } = await ctx.params;
    const id = decodeURIComponent(rawId);
    const found = await findMeldById(id);
    if (!found) return notFound(`Meld '${id}' not found`);
    return ok(found.meld, `filesystem:melds/${found.foundIn}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("meld_detail_failed", message);
  }
}
