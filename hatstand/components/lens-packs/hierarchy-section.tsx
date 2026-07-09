"use client";

import Link from "next/link";
import { useState } from "react";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorState,
} from "@/components/ui";
import {
  useLensPackLenses,
  type LensesQuery,
  type LensWithMeta,
} from "@/lib/hooks/use-lens-packs";
import type { LensPack } from "@/types";

type SortField = NonNullable<LensesQuery["sort"]>;
type SortDir = "asc" | "desc";

const DEFAULT_DIR: Record<SortField, SortDir> = {
  concept: "asc",
  layer: "asc",
  f1: "desc",
  training_samples: "desc",
};

const CATEGORY_VARIANT: Record<string, "muted" | "info" | "success" | "warning"> = {
  early: "info",
  mid: "muted",
  late: "warning",
};

export function HierarchySection({ pack }: { pack: LensPack }) {
  const [q, setQ] = useState("");
  const [layerFilter, setLayerFilter] = useState<string>("");
  const [sort, setSort] = useState<SortField>("f1");
  const [dir, setDir] = useState<SortDir>("desc");

  const parsedLayer = layerFilter === "" ? undefined : Number(layerFilter);
  const layerOpt =
    typeof parsedLayer === "number" && Number.isInteger(parsedLayer)
      ? parsedLayer
      : undefined;

  const lenses = useLensPackLenses(pack.id, {
    q: q.trim() || undefined,
    layer: layerOpt,
    sort,
    dir,
    limit: 50,
  });

  function onSortColumn(field: SortField) {
    if (field === sort) {
      setDir(dir === "asc" ? "desc" : "asc");
    } else {
      setSort(field);
      setDir(DEFAULT_DIR[field]);
    }
  }

  return (
    <Card id="hierarchy">
      <CardHeader>
        <CardTitle>Hierarchy</CardTitle>
        <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
          Per-concept lens inventory from <span className="font-mono">version_manifest.json</span>.
          Model layer (not ontological).
        </p>
      </CardHeader>
      <CardBody>
        {lenses.isLoading ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Loading lens inventory…
          </p>
        ) : lenses.isError ? (
          <ErrorState
            title="Failed to load lenses"
            message={lenses.error?.message}
            onRetry={() => lenses.refetch()}
          />
        ) : !lenses.data || lenses.data.entries.total === 0 ? (
          <EmptyState
            title="No lenses in manifest"
            description="version_manifest.json is missing or has an empty `lenses` map."
          />
        ) : (
          <>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search concept…"
                className="rounded border border-zinc-200 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900"
              />
              <input
                value={layerFilter}
                onChange={(e) => setLayerFilter(e.target.value)}
                placeholder="layer"
                inputMode="numeric"
                className="w-20 rounded border border-zinc-200 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900"
              />
              <span className="ml-auto text-zinc-500 dark:text-zinc-400">
                {lenses.data.entries.items.length} of{" "}
                {lenses.data.entries.total.toLocaleString()} shown
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-zinc-500 dark:text-zinc-400">
                    <SortHeader field="concept" sort={sort} dir={dir} onSort={onSortColumn}>
                      Concept
                    </SortHeader>
                    <SortHeader field="layer" sort={sort} dir={dir} onSort={onSortColumn}>
                      Layer
                    </SortHeader>
                    <th className="pr-3 pb-1 font-medium">Category</th>
                    <SortHeader field="f1" sort={sort} dir={dir} onSort={onSortColumn}>
                      F1
                    </SortHeader>
                    <SortHeader
                      field="training_samples"
                      sort={sort}
                      dir={dir}
                      onSort={onSortColumn}
                    >
                      Samples
                    </SortHeader>
                  </tr>
                </thead>
                <tbody>
                  {lenses.data.entries.items.map((l) => (
                    <LensRow
                      key={`${l.term}_L${l.selected_layer}`}
                      lens={l}
                      sourcePack={pack.concept_pack}
                    />
                  ))}
                </tbody>
              </table>
            </div>
            {lenses.data.entries.next_cursor ? (
              <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                Adjust filters to narrow further; full pagination via
                /api/admin/lens-packs/{"{id}"}/lenses?cursor=…
              </p>
            ) : null}
          </>
        )}
      </CardBody>
    </Card>
  );
}

function LensRow({
  lens,
  sourcePack,
}: {
  lens: LensWithMeta;
  sourcePack: string;
}) {
  const conceptHref =
    sourcePack && sourcePack !== "unknown"
      ? `/concept-packs/${encodeURIComponent(sourcePack)}/concepts/${encodeURIComponent(lens.term)}`
      : null;
  return (
    <tr className="border-t border-zinc-100 dark:border-zinc-800">
      <td className="py-1 pr-3 font-mono text-zinc-900 dark:text-zinc-100">
        {conceptHref ? (
          <Link
            href={conceptHref}
            className="text-sky-700 hover:underline dark:text-sky-400"
          >
            {lens.term}
          </Link>
        ) : (
          lens.term
        )}
      </td>
      <td className="py-1 pr-3 font-mono text-zinc-700 dark:text-zinc-300">
        {lens.selected_layer}
      </td>
      <td className="py-1 pr-3">
        {lens.category ? (
          <Badge variant={CATEGORY_VARIANT[lens.category] ?? "muted"}>
            {lens.category}
          </Badge>
        ) : (
          <span className="text-zinc-400">—</span>
        )}
      </td>
      <td className="py-1 pr-3 tabular-nums">
        <F1Cell f1={lens.training_metrics.test_f1} />
      </td>
      <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
        {lens.training_samples ?? "—"}
      </td>
    </tr>
  );
}

function F1Cell({ f1 }: { f1: number }) {
  const color =
    f1 >= 0.9
      ? "text-emerald-700 dark:text-emerald-300"
      : f1 >= 0.7
        ? "text-zinc-700 dark:text-zinc-300"
        : f1 >= 0.5
          ? "text-amber-700 dark:text-amber-300"
          : "text-rose-700 dark:text-rose-300";
  return <span className={color}>{f1.toFixed(3)}</span>;
}

function SortHeader({
  field,
  sort,
  dir,
  onSort,
  children,
}: {
  field: SortField;
  sort: SortField;
  dir: SortDir;
  onSort: (field: SortField) => void;
  children: React.ReactNode;
}) {
  const active = sort === field;
  const arrow = active ? (dir === "asc" ? " ↑" : " ↓") : " ↕";
  return (
    <th className="pr-3 pb-1 font-medium">
      <button
        type="button"
        onClick={() => onSort(field)}
        className={`inline-flex items-center transition-colors ${
          active
            ? "text-zinc-900 dark:text-zinc-100"
            : "hover:text-zinc-700 dark:hover:text-zinc-300"
        }`}
      >
        <span>{children}</span>
        <span className="ml-0.5 text-zinc-400 tabular-nums">{arrow}</span>
      </button>
    </th>
  );
}
