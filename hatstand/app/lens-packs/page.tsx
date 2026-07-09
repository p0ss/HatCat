"use client";

import Link from "next/link";
import { PageHeader } from "@/components/page-header";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorState,
} from "@/components/ui";
import { useLensPacks } from "@/lib/hooks/use-lens-packs";
import type {
  CalibrationStatus,
  LensPack,
  LensPackStatus,
} from "@/types";

const STATUS_VARIANT: Record<
  LensPackStatus,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  trained: "info",
  calibrating: "warning",
  uncalibrated: "warning",
  validated: "success",
  error: "error",
};

const CALIBRATION_VARIANT: Record<
  CalibrationStatus,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  complete: "success",
  partial: "warning",
  missing: "error",
};

function avgF1Across(pack: LensPack): number | null {
  const layers = Object.values(pack.aggregate_metrics?.avg_test_f1_per_layer ?? {});
  if (layers.length === 0) return null;
  return layers.reduce((a, b) => a + b, 0) / layers.length;
}

export default function LensPacksPage() {
  const { data, isLoading, isError, error, refetch } = useLensPacks();

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Lens Packs"
        description="Trained lens packs — coverage, calibration, simplexes, provenance."
      />
      <div className="mt-6">
        {isLoading ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading…</p>
        ) : isError ? (
          <ErrorState
            title="Failed to load lens packs"
            message={error?.message}
            onRetry={() => refetch()}
          />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No lens packs registered"
            description="Expected a registry at HatCatDev/src/lens_packs/.registry.json with one or more pack directories alongside it."
          />
        ) : (
          <ul className="grid gap-3 grid-cols-1 lg:grid-cols-2">
            {data.items.map((pack) => (
              <li key={pack.id}>
                <LensPackCard pack={pack} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function LensPackCard({ pack }: { pack: LensPack }) {
  const avg = avgF1Across(pack);
  return (
    <Link
      href={`/lens-packs/${encodeURIComponent(pack.id)}`}
      className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 rounded-lg"
    >
      <Card className="hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors">
        <CardHeader className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="truncate">{pack.id}</CardTitle>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400 truncate">
              {pack.substrate}
            </p>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <Badge variant={STATUS_VARIANT[pack.status]}>{pack.status}</Badge>
            <Badge variant={CALIBRATION_VARIANT[pack.calibration_status]}>
              calib: {pack.calibration_status}
            </Badge>
          </div>
        </CardHeader>
        <CardBody>
          <dl className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Concept pack</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100 truncate">
                {pack.concept_pack}@{pack.version}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Avg test F1</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100">
                {avg === null ? "—" : avg.toFixed(3)}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Based on</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100 truncate">
                {pack.based_on ?? "—"}
              </dd>
            </div>
          </dl>
        </CardBody>
      </Card>
    </Link>
  );
}
