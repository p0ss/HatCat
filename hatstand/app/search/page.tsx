"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
import { PageHeader } from "@/components/page-header";
import { Badge, EmptyState, ErrorState } from "@/components/ui";
import {
  FacetPanel,
  deriveResourceTypeCounts,
} from "@/components/search/facet-panel";
import { useSearch } from "@/lib/hooks/use-search";
import type { ResourceType, SearchDocument } from "@/types";

const TYPE_VARIANT: Record<
  ResourceType,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  model: "info",
  concept_pack: "info",
  concept: "muted",
  lens_pack: "success",
  lens: "muted",
  simplex: "muted",
  meld: "warning",
  run: "warning",
  doc: "muted",
};

function parseFilterParams(params: URLSearchParams): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const [key, value] of params.entries()) {
    const m = /^filter\[([^\]]+)\]$/.exec(key);
    if (!m) continue;
    (out[m[1]] ??= []).push(value);
  }
  return out;
}

export default function SearchPage() {
  return (
    <Suspense fallback={<SearchPageFallback />}>
      <SearchPageInner />
    </Suspense>
  );
}

function SearchPageFallback() {
  return (
    <div className="px-8 py-6">
      <PageHeader title="Search" description="Loading…" />
    </div>
  );
}

function SearchPageInner() {
  const router = useRouter();
  const params = useSearchParams();
  const q = params.get("q") ?? "";
  const type = (params.get("type") as ResourceType | null) ?? undefined;
  const filters = useMemo(() => parseFilterParams(params), [params]);

  const { data, isLoading, isError, error } = useSearch({
    q: q || undefined,
    type,
    filter: filters,
    limit: 50,
  });

  // Resource-type counts: re-query without `type` to get the breakdown.
  // Lightweight — same result for the same q+filter combination.
  const totalsQuery = useSearch({
    q: q || undefined,
    filter: filters,
    limit: 200,
  });
  const resourceTypeCounts = deriveResourceTypeCounts(totalsQuery.data);

  function setUrl(next: {
    q?: string;
    type?: ResourceType;
    filter?: Record<string, string[]>;
  }) {
    const usp = new URLSearchParams();
    if (next.q) usp.set("q", next.q);
    if (next.type) usp.set("type", next.type);
    if (next.filter) {
      for (const [k, vs] of Object.entries(next.filter)) {
        for (const v of vs) usp.append(`filter[${k}]`, v);
      }
    }
    const qs = usp.toString();
    router.push(`/search${qs ? `?${qs}` : ""}`);
  }

  function onSelectType(t: ResourceType | undefined) {
    setUrl({ q, type: t, filter: filters });
  }

  function onToggleFilter(facet: string, value: string) {
    const next = { ...filters };
    const current = next[facet] ?? [];
    next[facet] = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    if (next[facet].length === 0) delete next[facet];
    setUrl({ q, type, filter: next });
  }

  const subtitle = q
    ? `"${q}"${data ? ` — ${data.total} result${data.total === 1 ? "" : "s"}` : ""}`
    : data
      ? `${data.total} document${data.total === 1 ? "" : "s"} indexed`
      : "Universal search across all resources.";

  return (
    <div className="px-8 py-6">
      <PageHeader title="Search" description={subtitle} />
      <div className="mt-6 flex gap-6">
        <FacetPanel
          resourceTypeCounts={resourceTypeCounts}
          selectedType={type}
          onSelectType={onSelectType}
          facets={data?.facets ?? {}}
          selectedFilters={filters}
          onToggleFilter={onToggleFilter}
        />
        <section className="min-w-0 flex-1">
          {isLoading && !data ? (
            <p className="text-sm text-zinc-500 dark:text-zinc-400">Searching…</p>
          ) : isError ? (
            <ErrorState
              title="Search failed"
              message={error?.message}
            />
          ) : !data || data.items.length === 0 ? (
            <EmptyState
              title="No results"
              description={
                q
                  ? `No matches for "${q}". Try a broader query or different filters.`
                  : "Type something into the search bar above."
              }
            />
          ) : (
            <ul className="space-y-3">
              {data.items.map((item) => (
                <li key={item.id}>
                  <SearchResultRow item={item} />
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}

function SearchResultRow({ item }: { item: SearchDocument }) {
  return (
    <Link
      href={item.url}
      className="block rounded-lg border border-zinc-200 bg-white px-4 py-3 transition-colors hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
    >
      <div className="flex items-start gap-3">
        <Badge variant={TYPE_VARIANT[item.resource_type] ?? "muted"}>
          {item.resource_type.replace(/_/g, " ")}
        </Badge>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
            {item.title}
          </h3>
          {item.body_excerpt ? (
            <p className="mt-1 line-clamp-2 text-xs text-zinc-600 dark:text-zinc-400">
              {item.body_excerpt}
            </p>
          ) : null}
          <p className="mt-1 font-mono text-[10px] text-zinc-400 dark:text-zinc-500">
            {item.url}
          </p>
        </div>
      </div>
    </Link>
  );
}
