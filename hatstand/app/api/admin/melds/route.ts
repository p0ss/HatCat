// GET /api/admin/melds — list melds discovered in HatCatDev/melds/{pending,reference,applied}.
// Adapts on-disk JSON shape to the Meld type. State is derived from the folder
// the meld lives in, with a per-file override if a `state` field is present.

import { fail, ok } from "@/lib/server/api-helpers";
import { loadAllMelds } from "@/lib/server/melds";
import type { Meld, Page } from "@/types";

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const stateFilter = url.searchParams.get("state") ?? undefined;
    const sourceFilter = url.searchParams.get("source") ?? undefined;
    const targetPackFilter = url.searchParams.get("target_pack") ?? undefined;
    const q = url.searchParams.get("q")?.toLowerCase() ?? undefined;

    const items = await loadAllMelds();

    const filtered = items.filter((m) => {
      if (stateFilter && m.state !== stateFilter) return false;
      if (sourceFilter && m.source !== sourceFilter) return false;
      if (targetPackFilter && !m.target_pack.includes(targetPackFilter)) return false;
      if (q) {
        const haystack = [
          m.id,
          m.target_pack,
          ...m.candidates.map((c) => c.term ?? ""),
          ...m.candidates.map((c) => c.parent ?? ""),
        ]
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });

    filtered.sort((a, b) => {
      if (a.updated_at && b.updated_at)
        return b.updated_at.localeCompare(a.updated_at);
      return a.id.localeCompare(b.id);
    });

    const page: Page<Meld> = { items: filtered, total: filtered.length };
    return ok(page, "filesystem:melds");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("melds_list_failed", message);
  }
}
