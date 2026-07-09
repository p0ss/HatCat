"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { PageHeader } from "@/components/page-header";
import { MeldStateBadge } from "@/components/melds/meld-state-badge";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  EmptyState,
  ErrorState,
} from "@/components/ui";
import { useMelds, type MeldFilters } from "@/lib/hooks/use-melds";
import type { Meld, MeldSource, MeldState, ProtectionLevel } from "@/types";

const STATE_ORDER: MeldState[] = [
  "tender",
  "review",
  "authorise",
  "commit",
  "evaluate",
  "rejected",
];

const STATE_LABEL: Record<MeldState, string> = {
  tender: "Tender",
  review: "Review",
  authorise: "Authorise",
  commit: "Commit",
  evaluate: "Evaluate",
  rejected: "Rejected",
};

const SOURCES: MeldSource[] = ["manual", "be_discovery", "cat", "cross_be", "external"];

const PROTECTION_VARIANT: Record<
  ProtectionLevel,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  open: "muted",
  guarded: "warning",
  sealed: "error",
};

function relativeTime(iso: string): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const diffMs = Date.now() - t;
  const sec = Math.round(diffMs / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day < 30) return `${day}d ago`;
  const mo = Math.round(day / 30);
  if (mo < 12) return `${mo}mo ago`;
  return `${Math.round(mo / 12)}y ago`;
}

export default function MeldsPage() {
  const [stateFilter, setStateFilter] = useState<MeldState | "">("");
  const [sourceFilter, setSourceFilter] = useState<MeldSource | "">("");
  const [targetPackFilter, setTargetPackFilter] = useState("");
  const [q, setQ] = useState("");
  const [collapsed, setCollapsed] = useState<Record<MeldState, boolean>>({
    tender: false,
    review: false,
    authorise: false,
    commit: false,
    evaluate: false,
    rejected: false,
  });

  const filters: MeldFilters | undefined = useMemo(() => {
    const f: MeldFilters = {};
    if (stateFilter) f.state = stateFilter;
    if (sourceFilter) f.source = sourceFilter;
    if (targetPackFilter.trim()) f.target_pack = targetPackFilter.trim();
    if (q.trim()) f.q = q.trim();
    return Object.keys(f).length ? f : undefined;
  }, [stateFilter, sourceFilter, targetPackFilter, q]);

  const { data, isLoading, isError, error, refetch } = useMelds(filters);

  const grouped = useMemo(() => {
    const out: Record<MeldState, Meld[]> = {
      tender: [],
      review: [],
      authorise: [],
      commit: [],
      evaluate: [],
      rejected: [],
    };
    for (const m of data?.items ?? []) out[m.state].push(m);
    return out;
  }, [data]);

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Melds"
        description="ASK TRACE pipeline — Tender → Review → Authorise → Commit → Evaluate."
      />

      <div className="mt-4 flex flex-wrap gap-2 items-end">
        <FilterField label="State">
          <select
            value={stateFilter}
            onChange={(e) => setStateFilter(e.target.value as MeldState | "")}
            className="filter-input"
          >
            <option value="">All</option>
            {STATE_ORDER.map((s) => (
              <option key={s} value={s}>{STATE_LABEL[s]}</option>
            ))}
          </select>
        </FilterField>
        <FilterField label="Source">
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value as MeldSource | "")}
            className="filter-input"
          >
            <option value="">All</option>
            {SOURCES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </FilterField>
        <FilterField label="Target pack">
          <input
            value={targetPackFilter}
            onChange={(e) => setTargetPackFilter(e.target.value)}
            placeholder="first-light"
            className="filter-input"
          />
        </FilterField>
        <FilterField label="Search">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="id or candidate"
            className="filter-input"
          />
        </FilterField>
        <style jsx>{`
          .filter-input {
            font-size: 0.75rem;
            padding: 0.375rem 0.5rem;
            border-radius: 0.375rem;
            border: 1px solid rgb(228 228 231);
            background: white;
            min-width: 9rem;
          }
          :global(.dark) .filter-input {
            background: rgb(24 24 27);
            border-color: rgb(63 63 70);
            color: rgb(244 244 245);
          }
        `}</style>
      </div>

      <div className="mt-6">
        {isLoading ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading…</p>
        ) : isError ? (
          <ErrorState
            title="Failed to load melds"
            message={error?.message}
            onRetry={() => refetch()}
          />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No melds found"
            description="Expected meld files under HatCatDev/melds/{pending,reference,applied}/."
          />
        ) : (
          <div className="grid gap-6">
            {STATE_ORDER.map((state) => {
              const items = grouped[state];
              const isCollapsed = collapsed[state];
              return (
                <section key={state}>
                  <button
                    type="button"
                    onClick={() =>
                      setCollapsed((prev) => ({ ...prev, [state]: !prev[state] }))
                    }
                    className="flex items-center gap-2 mb-2 text-left"
                  >
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {isCollapsed ? "▶" : "▼"}
                    </span>
                    <MeldStateBadge state={state} />
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {items.length}
                    </span>
                  </button>
                  {!isCollapsed && (
                    items.length === 0 ? (
                      <p className="ml-6 text-xs text-zinc-500 dark:text-zinc-400">
                        No melds in this state.
                      </p>
                    ) : (
                      <ul className="grid gap-2 grid-cols-1 lg:grid-cols-2">
                        {items.map((m) => (
                          <li key={m.id}>
                            <MeldCard meld={m} />
                          </li>
                        ))}
                      </ul>
                    )
                  )}
                </section>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function FilterField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
        {label}
      </span>
      {children}
    </label>
  );
}

function MeldCard({ meld }: { meld: Meld }) {
  return (
    <Link
      href={`/melds/${encodeURIComponent(meld.id)}`}
      className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 rounded-lg"
    >
      <Card className="hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors">
        <CardHeader className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="truncate">{meld.id}</CardTitle>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400 truncate">
              → {meld.target_pack}
            </p>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <MeldStateBadge state={meld.state} />
            <Badge variant="info">{meld.source}</Badge>
          </div>
        </CardHeader>
        <CardBody>
          <dl className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Candidates</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100">
                {meld.candidates.length}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Protection</dt>
              <dd className="mt-0.5">
                <Badge variant={PROTECTION_VARIANT[meld.protection_level]}>
                  {meld.protection_level}
                </Badge>
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Updated</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100">
                {relativeTime(meld.updated_at)}
              </dd>
            </div>
          </dl>
        </CardBody>
      </Card>
    </Link>
  );
}
