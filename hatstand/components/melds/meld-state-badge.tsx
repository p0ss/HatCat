import { Badge } from "@/components/ui";
import type { MeldState } from "@/types";

const STATE_VARIANT: Record<
  MeldState,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  tender: "muted",
  review: "info",
  authorise: "warning",
  commit: "success",
  evaluate: "success",
  rejected: "error",
};

export function MeldStateBadge({ state }: { state: MeldState }) {
  return <Badge variant={STATE_VARIANT[state]}>{state}</Badge>;
}
