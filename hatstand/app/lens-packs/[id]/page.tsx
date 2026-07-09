"use client";

import Link from "next/link";
import { use } from "react";
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
import { useLensPack } from "@/lib/hooks/use-lens-packs";
import { CalibrationSection } from "@/components/lens-packs/calibration-section";
import { HierarchySection } from "@/components/lens-packs/hierarchy-section";
import { SimplexSection } from "@/components/lens-packs/simplex-section";
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

export default function LensPackDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: rawId } = use(params);
  const id = decodeURIComponent(rawId);
  const { data, isLoading, isError, error, refetch } = useLensPack(id);

  if (isLoading) {
    return (
      <div className="px-8 py-6">
        <PageHeader title={id} description="Loading…" />
      </div>
    );
  }

  if (isError) {
    const isNotFound =
      (error as { code?: string } | undefined)?.code === "not_found";
    if (isNotFound) {
      return (
        <div className="px-8 py-6">
          <PageHeader title={id} />
          <div className="mt-6">
            <EmptyState
              title="Lens pack not found"
              description={`No directory found at HatCatDev/src/lens_packs/${id}.`}
              action={
                <Link
                  href="/lens-packs"
                  className="text-sm font-medium text-sky-700 hover:underline dark:text-sky-400"
                >
                  Back to lens packs
                </Link>
              }
            />
          </div>
        </div>
      );
    }
    return (
      <div className="px-8 py-6">
        <PageHeader title={id} />
        <div className="mt-6">
          <ErrorState
            title="Failed to load lens pack"
            message={error?.message}
            onRetry={() => refetch()}
          />
        </div>
      </div>
    );
  }

  if (!data) return null;
  return <LensPackDetail pack={data} />;
}

function LensPackDetail({ pack }: { pack: LensPack }) {
  const layerEntries = Object.entries(pack.aggregate_metrics.avg_test_f1_per_layer)
    .map(([layer, f1]) => ({ layer: Number(layer), f1 }))
    .sort((a, b) => a.layer - b.layer);

  const sectionLink =
    "block text-xs font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100";

  return (
    <div className="px-8 py-6">
      <PageHeader
        title={pack.id}
        description={`${pack.substrate} · ${pack.concept_pack}@${pack.version}`}
        actions={
          <>
            <Badge variant={STATUS_VARIANT[pack.status]}>{pack.status}</Badge>
            <Badge variant={CALIBRATION_VARIANT[pack.calibration_status]}>
              calib: {pack.calibration_status}
            </Badge>
          </>
        }
      />

      <nav className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <a href="#metrics" className={sectionLink}>Aggregate metrics</a>
        <a href="#hierarchy" className={sectionLink}>Hierarchy</a>
        <a href="#simplexes" className={sectionLink}>Simplexes</a>
        <a href="#calibration" className={sectionLink}>Calibration</a>
        <a href="#provenance" className={sectionLink}>Provenance</a>
      </nav>

      <div className="mt-6 grid gap-4">
        <Card id="metrics">
          <CardHeader>
            <CardTitle>Aggregate metrics</CardTitle>
          </CardHeader>
          <CardBody>
            {layerEntries.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No per-layer metrics available.
              </p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                    <th className="font-medium pb-2">Layer</th>
                    <th className="font-medium pb-2">Avg test F1</th>
                  </tr>
                </thead>
                <tbody>
                  {layerEntries.map(({ layer, f1 }) => (
                    <tr key={layer} className="border-t border-zinc-100 dark:border-zinc-800">
                      <td className="py-1.5 font-mono text-xs text-zinc-700 dark:text-zinc-300">
                        layer{layer}
                      </td>
                      <td className="py-1.5 font-medium text-zinc-900 dark:text-zinc-100">
                        {f1.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>

        <HierarchySection pack={pack} />

        <SimplexSection pack={pack} />

        <CalibrationSection pack={pack} />

        <Card id="provenance">
          <CardHeader>
            <CardTitle>Provenance</CardTitle>
          </CardHeader>
          <CardBody>
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-xs text-zinc-500 dark:text-zinc-400">Based on</dt>
                <dd className="mt-0.5 font-mono text-xs text-zinc-900 dark:text-zinc-100 break-all">
                  {pack.based_on ?? "—"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500 dark:text-zinc-400">Registry path</dt>
                <dd className="mt-0.5 font-mono text-xs text-zinc-900 dark:text-zinc-100 break-all">
                  {pack.registry_path ?? "—"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500 dark:text-zinc-400">Created at</dt>
                <dd className="mt-0.5 text-xs text-zinc-900 dark:text-zinc-100">
                  {pack.created_at || "—"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-zinc-500 dark:text-zinc-400">Updated at</dt>
                <dd className="mt-0.5 text-xs text-zinc-900 dark:text-zinc-100">
                  {pack.updated_at || "—"}
                </dd>
              </div>
            </dl>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
