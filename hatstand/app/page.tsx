"use client";

import { PageHeader } from "@/components/page-header";
import { Badge, Card, CardBody, CardHeader, CardTitle } from "@/components/ui";
import { useEnv } from "@/lib/hooks/use-env";
import { useRegistry } from "@/lib/hooks/use-registry";

function humanizeBytes(bytes: number | undefined): string {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(value >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export default function DashboardPage() {
  const env = useEnv();
  const registry = useRegistry();

  // Count unique substrates referenced by lens packs as a proxy for "models in
  // use" — cheaper than scanning HF cache for individual model dirs and more
  // meaningful (the substrates we *care* about are the ones backing packs).
  const substrateCount = registry.data
    ? new Set(registry.data.lens_packs.map((p) => p.substrate)).size
    : undefined;

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Dashboard"
        description="HatCat lifecycle at a glance — packs, runs, melds, recent activity."
      />
      <div className="mt-6 grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
        <StatCard
          title="Substrates"
          hint="Models backing lens packs."
          value={substrateCount}
          loading={registry.isLoading}
          error={registry.isError}
          footer={
            env.data?.hf_cache.size_bytes
              ? `HF cache: ${humanizeBytes(env.data.hf_cache.size_bytes)}`
              : undefined
          }
        />
        <StatCard
          title="Concept Packs"
          hint="Ontology revisions in flight."
          value={registry.data?.concept_packs.length}
          loading={registry.isLoading}
          error={registry.isError}
        />
        <StatCard
          title="Lens Packs"
          hint="Status / coverage / calibration."
          value={registry.data?.lens_packs.length}
          loading={registry.isLoading}
          error={registry.isError}
        />
        <PlaceholderCard
          title="Active Runs"
          hint="Live training, calibration, and eval jobs."
        />
        <PlaceholderCard
          title="Recent Melds"
          hint="ASK TRACE pipeline state."
        />
        <PlaceholderCard
          title="Search"
          hint="Universal index across all resources."
        />
      </div>
      <p className="mt-8 text-xs text-zinc-500 dark:text-zinc-400">
        Substrates / Concept Packs / Lens Packs read live from HatCatDev. Other
        cards become live once their slice indexers land.
      </p>
    </div>
  );
}

function StatCard({
  title,
  hint,
  value,
  loading,
  error,
  footer,
}: {
  title: string;
  hint: string;
  value: number | undefined;
  loading: boolean;
  error: boolean;
  footer?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardBody>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</p>
        <div className="mt-3 flex items-baseline gap-2">
          {loading ? (
            <Badge variant="muted">loading…</Badge>
          ) : error ? (
            <Badge variant="error">error</Badge>
          ) : (
            <span className="text-3xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
              {value ?? 0}
            </span>
          )}
        </div>
        {footer ? (
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            {footer}
          </p>
        ) : null}
      </CardBody>
    </Card>
  );
}

function PlaceholderCard({ title, hint }: { title: string; hint: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardBody>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</p>
        <div className="mt-3 h-12 rounded-md border border-dashed border-zinc-300 dark:border-zinc-700" />
      </CardBody>
    </Card>
  );
}
