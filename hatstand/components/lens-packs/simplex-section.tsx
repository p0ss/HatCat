"use client";

import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorState,
} from "@/components/ui";
import { useLensPackSimplexes } from "@/lib/hooks/use-lens-packs";
import type { LensPack, Simplex, SimplexPole } from "@/types";

const POLE_VARIANT: Record<SimplexPole["name"], "success" | "muted" | "error"> = {
  positive: "success",
  neutral: "muted",
  negative: "error",
};

export function SimplexSection({ pack }: { pack: LensPack }) {
  void pack; // pack passed for symmetry; component uses pack.id via hook below
  const { data, isLoading, isError, error, refetch } = useLensPackSimplexes(
    pack.id,
  );

  return (
    <Card id="simplexes">
      <CardHeader>
        <CardTitle>Simplex inventory</CardTitle>
        <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
          Three-pole simplexes from{" "}
          <span className="font-mono">simplex/_definitions.json</span>.
          Per-pole training metrics aren&apos;t carried in the pack — see source
          run for those.
        </p>
      </CardHeader>
      <CardBody>
        {isLoading ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Loading simplex inventory…
          </p>
        ) : isError ? (
          <ErrorState
            title="Failed to load simplexes"
            message={error?.message}
            onRetry={() => refetch()}
          />
        ) : !data || data.simplexes.length === 0 ? (
          <EmptyState
            title="No simplexes in pack"
            description="This pack has no simplex/_definitions.json or pack_info.simplexes is empty."
          />
        ) : (
          <>
            <InventoryMeta info={data.info} count={data.simplexes.length} />
            <div className="mt-4 grid gap-3 grid-cols-1 lg:grid-cols-2">
              {data.simplexes.map((s) => (
                <SimplexCard key={s.id} simplex={s} />
              ))}
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function InventoryMeta({
  info,
  count,
}: {
  info: {
    format?: string;
    source?: string;
    trained_with?: string;
    definitions_version?: string;
    definitions_note?: string;
  };
  count: number;
}) {
  const items: Array<[string, string | undefined]> = [
    ["Count", String(count)],
    ["Format", info.format],
    ["Defs version", info.definitions_version],
    ["Source run", info.source],
  ];
  return (
    <div className="rounded border border-zinc-200 bg-zinc-50 px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900/50">
      <dl className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-4">
        {items
          .filter(([, v]) => v != null && v !== "")
          .map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <dt className="text-zinc-500 dark:text-zinc-400">{k}:</dt>
              <dd className="min-w-0 truncate font-mono text-zinc-900 dark:text-zinc-100">
                {v}
              </dd>
            </div>
          ))}
      </dl>
      {info.definitions_note ? (
        <p className="mt-1 text-zinc-500 dark:text-zinc-400">
          {info.definitions_note}
        </p>
      ) : null}
    </div>
  );
}

function SimplexCard({ simplex }: { simplex: Simplex }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-start justify-between gap-2">
        <h4 className="text-sm font-medium text-zinc-900 dark:text-zinc-100 font-mono">
          {simplex.dimension}
        </h4>
        <Badge variant="muted">{simplex.poles.length}-pole</Badge>
      </div>
      <ul className="mt-2 space-y-1.5">
        {simplex.poles.map((p) => (
          <li key={p.name} className="flex items-start gap-2">
            <Badge variant={POLE_VARIANT[p.name]} className="shrink-0">
              {p.name}
            </Badge>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                {p.label ?? "—"}
                {typeof p.test_f1 === "number" ? (
                  <span className="ml-2 font-mono text-zinc-500 dark:text-zinc-400">
                    F1 {p.test_f1.toFixed(3)}
                  </span>
                ) : null}
              </p>
              {p.definition ? (
                <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
                  {p.definition}
                </p>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
