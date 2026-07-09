"use client";

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
  useLensPackCalibration,
  type CalibrationQuery,
} from "@/lib/hooks/use-lens-packs";
import type {
  CalibrationCycle,
  ConceptCalibration,
  LensPack,
} from "@/types";

type Props = { pack: LensPack };

type SortField = NonNullable<CalibrationQuery["sort"]>;
type SortDir = "asc" | "desc";

// Concept defaults to ascending (A→Z); rates and means to descending (worst first).
const DEFAULT_DIR: Record<SortField, SortDir> = {
  concept: "asc",
  self_mean: "desc",
  cross_fire_rate: "desc",
  noise_fire_rate: "desc",
};

export function CalibrationSection({ pack }: Props) {
  const [overFirersOnly, setOverFirersOnly] = useState(true);
  const [sort, setSort] = useState<SortField>("cross_fire_rate");
  const [dir, setDir] = useState<SortDir>("desc");

  const calib = useLensPackCalibration(pack.id, {
    overFirersOnly,
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

  if (calib.isLoading) {
    return (
      <Card id="calibration">
        <CardHeader>
          <CardTitle>Calibration</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Loading calibration data…
          </p>
        </CardBody>
      </Card>
    );
  }

  if (calib.isError) {
    const err = calib.error as { code?: string } | undefined;
    return (
      <Card id="calibration">
        <CardHeader>
          <CardTitle>Calibration</CardTitle>
        </CardHeader>
        <CardBody>
          {err?.code === "not_found" ? (
            <EmptyState
              title="No calibration on disk"
              description="The calibration cycle hasn't been run for this pack yet."
            />
          ) : (
            <ErrorState
              title="Failed to load calibration"
              message={calib.error?.message}
              onRetry={() => calib.refetch()}
            />
          )}
        </CardBody>
      </Card>
    );
  }

  const data = calib.data;
  if (!data) return null;
  const { summary, cycles, has_noise_track, mode } = data;

  return (
    <Card id="calibration">
      <CardHeader className="flex items-start justify-between gap-2">
        <div>
          <CardTitle>Calibration</CardTitle>
          <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
            {summary.total_entries.toLocaleString()} (concept × layer) entries
            {mode ? ` · mode: ${mode}` : ""}
            {has_noise_track ? " · has noise track" : ""}
          </p>
        </div>
      </CardHeader>
      <CardBody className="space-y-5">
        {/* Summary row */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <Stat
            label="Cross over-firers"
            value={summary.over_firers}
            total={summary.total_entries}
            tone={summary.over_firers / summary.total_entries > 0.4 ? "error" : "warning"}
          />
          {has_noise_track ? (
            <Stat
              label="Noise over-firers"
              value={summary.noise_over_firers ?? 0}
              total={summary.total_entries}
              tone={
                (summary.noise_over_firers ?? 0) / summary.total_entries > 0.3
                  ? "warning"
                  : "muted"
              }
            />
          ) : null}
          <Stat
            label="Threshold"
            valueText={`fire_rate > ${summary.threshold}`}
            tone="muted"
          />
        </div>

        {/* Per-layer breakdown */}
        <PerLayerBreakdown summary={summary} hasNoise={has_noise_track} />

        {/* Cycle progression */}
        {cycles.length > 0 ? <CyclesTable cycles={cycles} /> : null}

        {/* Entries table */}
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
              Per-concept entries
            </h4>
            <label className="flex items-center gap-1.5 text-xs">
              <input
                type="checkbox"
                checked={overFirersOnly}
                onChange={(e) => setOverFirersOnly(e.target.checked)}
              />
              <span className="text-zinc-600 dark:text-zinc-400">
                Over-firers only
              </span>
            </label>
          </div>
          <EntryTable
            entries={data.entries.items}
            hasNoise={has_noise_track}
            sort={sort}
            dir={dir}
            onSort={onSortColumn}
          />
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            Showing {data.entries.items.length} of{" "}
            {data.entries.total.toLocaleString()} matching entries.
            {data.entries.next_cursor
              ? " Adjust filters to narrow further; full pagination via /api/admin/lens-packs/{id}/calibration?cursor=…"
              : ""}
          </p>
        </div>
      </CardBody>
    </Card>
  );
}

function Stat({
  label,
  value,
  total,
  valueText,
  tone,
}: {
  label: string;
  value?: number;
  total?: number;
  valueText?: string;
  tone: "muted" | "warning" | "error";
}) {
  const toneColor =
    tone === "error"
      ? "text-rose-700 dark:text-rose-300"
      : tone === "warning"
        ? "text-amber-700 dark:text-amber-300"
        : "text-zinc-900 dark:text-zinc-100";
  return (
    <div>
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className={`mt-0.5 text-base font-semibold tabular-nums ${toneColor}`}>
        {valueText ??
          (typeof value === "number" && typeof total === "number"
            ? `${value.toLocaleString()} / ${total.toLocaleString()}`
            : value?.toLocaleString() ?? "—")}
      </dd>
      {typeof value === "number" && typeof total === "number" && total > 0 ? (
        <dd className="text-xs text-zinc-500 dark:text-zinc-400">
          {((100 * value) / total).toFixed(1)}%
        </dd>
      ) : null}
    </div>
  );
}

function PerLayerBreakdown({
  summary,
  hasNoise,
}: {
  summary: NonNullable<ReturnType<typeof useLensPackCalibration>["data"]>["summary"];
  hasNoise: boolean;
}) {
  return (
    <div>
      <h4 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Per-layer over-firers
      </h4>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr className="text-left text-zinc-500 dark:text-zinc-400">
              <th className="pr-3 pb-1 font-medium">Layer</th>
              <th className="pr-3 pb-1 font-medium">Total</th>
              <th className="pr-3 pb-1 font-medium">Cross over</th>
              {hasNoise ? (
                <th className="pr-3 pb-1 font-medium">Noise over</th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {summary.per_layer.map((pl) => {
              const cratio = pl.total > 0 ? pl.over_firers / pl.total : 0;
              const nratio =
                pl.total > 0 && pl.noise_over_firers !== undefined
                  ? pl.noise_over_firers / pl.total
                  : undefined;
              return (
                <tr
                  key={pl.layer}
                  className="border-t border-zinc-100 dark:border-zinc-800"
                >
                  <td className="py-1 pr-3 font-mono text-zinc-700 dark:text-zinc-300">
                    L{pl.layer}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                    {pl.total}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-zinc-900 dark:text-zinc-100">
                    {pl.over_firers}
                    <span className="ml-1 text-zinc-500 dark:text-zinc-400">
                      ({(cratio * 100).toFixed(0)}%)
                    </span>
                  </td>
                  {hasNoise ? (
                    <td className="py-1 pr-3 tabular-nums text-zinc-900 dark:text-zinc-100">
                      {pl.noise_over_firers ?? "—"}
                      {nratio !== undefined ? (
                        <span className="ml-1 text-zinc-500 dark:text-zinc-400">
                          ({(nratio * 100).toFixed(0)}%)
                        </span>
                      ) : null}
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CyclesTable({ cycles }: { cycles: CalibrationCycle[] }) {
  // Detect non-monotonic regression — well_calibrated dropping between cycles.
  const regression = cycles.some(
    (c, i) => i > 0 && c.well_calibrated < cycles[i - 1].well_calibrated,
  );
  return (
    <div>
      <h4 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Cycle progression
        {regression ? (
          <Badge variant="warning" className="ml-2">
            non-monotonic
          </Badge>
        ) : null}
      </h4>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-zinc-500 dark:text-zinc-400">
              <th className="pr-3 pb-1 font-medium">Cycle</th>
              <th className="pr-3 pb-1 font-medium">Well</th>
              <th className="pr-3 pb-1 font-medium">Under</th>
              <th className="pr-3 pb-1 font-medium">Over</th>
              <th className="pr-3 pb-1 font-medium">Top-k rate</th>
              <th className="pr-3 pb-1 font-medium">Boosted</th>
              <th className="pr-3 pb-1 font-medium">Avg improvement</th>
            </tr>
          </thead>
          <tbody>
            {cycles.map((c) => (
              <tr
                key={c.cycle}
                className="border-t border-zinc-100 dark:border-zinc-800"
              >
                <td className="py-1 pr-3 font-mono text-zinc-700 dark:text-zinc-300">
                  {c.cycle}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-900 dark:text-zinc-100">
                  {c.well_calibrated.toLocaleString()}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {c.under_firing.toLocaleString()}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {c.over_firing.toLocaleString()}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {c.avg_in_top_k_rate?.toFixed(3) ?? "—"}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {c.finetune?.lenses_boosted.toLocaleString() ?? "—"}
                </td>
                <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {c.finetune?.avg_improvement.toFixed(3) ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EntryTable({
  entries,
  hasNoise,
  sort,
  dir,
  onSort,
}: {
  entries: ConceptCalibration[];
  hasNoise: boolean;
  sort: SortField;
  dir: SortDir;
  onSort: (field: SortField) => void;
}) {
  if (entries.length === 0) {
    return (
      <p className="rounded border border-dashed border-zinc-300 p-3 text-xs text-zinc-500 dark:border-zinc-700 dark:text-zinc-500">
        No entries match the current filters.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-zinc-500 dark:text-zinc-400">
            <SortHeader field="concept" sort={sort} dir={dir} onSort={onSort}>
              Concept
            </SortHeader>
            <th className="pr-3 pb-1 font-medium">L</th>
            <SortHeader field="self_mean" sort={sort} dir={dir} onSort={onSort}>
              self_mean
            </SortHeader>
            <SortHeader
              field="cross_fire_rate"
              sort={sort}
              dir={dir}
              onSort={onSort}
            >
              cross_fire
            </SortHeader>
            {hasNoise ? (
              <SortHeader
                field="noise_fire_rate"
                sort={sort}
                dir={dir}
                onSort={onSort}
              >
                noise_fire
              </SortHeader>
            ) : null}
            <th className="pr-3 pb-1 font-medium">samples</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr
              key={`${e.concept}_L${e.layer}`}
              className="border-t border-zinc-100 dark:border-zinc-800"
            >
              <td className="py-1 pr-3 font-mono text-zinc-900 dark:text-zinc-100">
                {e.concept}
              </td>
              <td className="py-1 pr-3 font-mono text-zinc-700 dark:text-zinc-300">
                {e.layer}
              </td>
              <td className="py-1 pr-3 tabular-nums text-zinc-700 dark:text-zinc-300">
                {e.self_mean.toFixed(3)}
              </td>
              <td className="py-1 pr-3 tabular-nums">
                <RateCell rate={e.cross_fire_rate} />
              </td>
              {hasNoise ? (
                <td className="py-1 pr-3 tabular-nums">
                  {typeof e.noise_fire_rate === "number" ? (
                    <RateCell rate={e.noise_fire_rate} />
                  ) : (
                    <span className="text-zinc-400">—</span>
                  )}
                </td>
              ) : null}
              <td className="py-1 pr-3 tabular-nums text-zinc-500 dark:text-zinc-400">
                {e.n_self_samples}/{e.n_cross_samples}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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
  const arrow = active ? (dir === "asc" ? " ↑" : " ↓") : "";
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
        <span className="ml-0.5 text-zinc-400 tabular-nums">
          {active ? arrow : " ↕"}
        </span>
      </button>
    </th>
  );
}

function RateCell({ rate }: { rate: number }) {
  const color =
    rate >= 0.5
      ? "text-rose-700 dark:text-rose-300"
      : rate >= 0.3
        ? "text-amber-700 dark:text-amber-300"
        : rate >= 0.1
          ? "text-zinc-700 dark:text-zinc-300"
          : "text-zinc-500 dark:text-zinc-500";
  return <span className={color}>{rate.toFixed(3)}</span>;
}
