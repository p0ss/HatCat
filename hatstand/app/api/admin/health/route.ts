// GET /api/admin/health — minimal liveness/readiness check.
// No HatCatDev FS reads here; this is the cheapest "are you alive" endpoint.

import { ok, fail } from "@/lib/server/api-helpers";
import type { HealthStatus } from "@/types";

export async function GET() {
  try {
    const data: HealthStatus = {
      status: "ok",
      version: "0.1.0",
      uptime_seconds: process.uptime(),
    };
    return ok(data, "hatstand");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("health_failed", message);
  }
}
