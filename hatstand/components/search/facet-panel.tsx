"use client";

import type { FacetCounts, ResourceType, SearchResponse } from "@/types";

const RESOURCE_LABEL: Record<ResourceType, string> = {
  model: "Models",
  concept_pack: "Concept packs",
  concept: "Concepts",
  lens_pack: "Lens packs",
  lens: "Lenses",
  simplex: "Simplexes",
  meld: "Melds",
  run: "Runs",
  doc: "Docs",
};

const FACET_LABEL: Record<string, string> = {
  layer: "Layer",
  domain: "Domain",
  safety_tag: "Safety",
  family: "Family",
  cached: "Cached",
  status: "Status",
  calibration_status: "Calibration",
  substrate: "Substrate",
  state: "State",
  source: "Source",
  target_pack: "Target",
  protection_level: "Protection",
  folder: "Folder",
};

export type FacetPanelProps = {
  // Counts of results per resource_type. Computed by the page since the
  // search response only knows about the currently-selected type.
  resourceTypeCounts: Array<{ type: ResourceType; count: number }>;
  selectedType: ResourceType | undefined;
  onSelectType: (type: ResourceType | undefined) => void;
  // Facet aggregations for the currently-selected type.
  facets: FacetCounts;
  selectedFilters: Record<string, string[]>;
  onToggleFilter: (facet: string, value: string) => void;
};

export function FacetPanel(props: FacetPanelProps) {
  const {
    resourceTypeCounts,
    selectedType,
    onSelectType,
    facets,
    selectedFilters,
    onToggleFilter,
  } = props;

  return (
    <aside className="w-56 shrink-0 space-y-5 text-sm">
      <FacetSection title="Type">
        <ul className="space-y-0.5">
          <li>
            <FacetRow
              label="All"
              count={resourceTypeCounts.reduce((s, x) => s + x.count, 0)}
              active={selectedType === undefined}
              onClick={() => onSelectType(undefined)}
            />
          </li>
          {resourceTypeCounts.map(({ type, count }) => (
            <li key={type}>
              <FacetRow
                label={RESOURCE_LABEL[type] ?? type}
                count={count}
                active={selectedType === type}
                onClick={() =>
                  onSelectType(selectedType === type ? undefined : type)
                }
              />
            </li>
          ))}
        </ul>
      </FacetSection>

      {Object.entries(facets).map(([key, values]) => (
        <FacetSection key={key} title={FACET_LABEL[key] ?? key}>
          <ul className="space-y-0.5 max-h-56 overflow-auto">
            {values?.slice(0, 20).map((fv) => {
              const checked =
                selectedFilters[key]?.includes(fv.value) ?? false;
              return (
                <li key={fv.value}>
                  <FacetRow
                    label={fv.value}
                    count={fv.count}
                    active={checked}
                    onClick={() => onToggleFilter(key, fv.value)}
                  />
                </li>
              );
            })}
          </ul>
        </FacetSection>
      ))}
    </aside>
  );
}

function FacetSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {title}
      </h3>
      {children}
    </div>
  );
}

function FacetRow({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-center justify-between rounded px-2 py-1 text-left text-xs transition-colors ${
        active
          ? "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-200"
          : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900"
      }`}
    >
      <span className="truncate">{label}</span>
      <span className="ml-2 shrink-0 tabular-nums text-zinc-500 dark:text-zinc-500">
        {count}
      </span>
    </button>
  );
}

// Helper so the page can compute resource-type counts from any search result
// snapshot. This walks the items list, not the facets, since `facets` only
// covers the currently-filtered set.
export function deriveResourceTypeCounts(
  result: SearchResponse | undefined,
): Array<{ type: ResourceType; count: number }> {
  if (!result) return [];
  const counts = new Map<ResourceType, number>();
  for (const item of result.items) {
    counts.set(item.resource_type, (counts.get(item.resource_type) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count);
}
