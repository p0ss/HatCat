"use client";

import Link from "next/link";
import { use, useState } from "react";
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
import { HierarchyBrowser } from "@/components/concept-packs/hierarchy-browser";
import {
  useConceptPack,
  useConceptPackConcepts,
} from "@/lib/hooks/use-concept-packs";
import type { Concept } from "@/types";

const LIST_LIMIT = 200;
const KNOWN_LAYERS = [0, 1, 2, 3, 4, 5, 6];

export default function ConceptPackDetailPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name: rawName } = use(params);
  const name = decodeURIComponent(rawName);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [query, setQuery] = useState("");

  const packQ = useConceptPack(name);

  const conceptsQ = useConceptPackConcepts(name, {
    layer: selectedLayer ?? undefined,
    q: query.trim() || undefined,
    limit: LIST_LIMIT,
  });

  if (packQ.isLoading) {
    return (
      <div className="px-8 py-6">
        <PageHeader title={name} description="Loading…" />
      </div>
    );
  }

  if (packQ.isError) {
    const isNotFound =
      (packQ.error as { code?: string } | undefined)?.code === "not_found";
    if (isNotFound) {
      return (
        <div className="px-8 py-6">
          <PageHeader title={name} />
          <div className="mt-6">
            <EmptyState
              title="Concept pack not found"
              description={`No directory found at HatCatDev/concept_packs/${name}.`}
              action={
                <Link
                  href="/concept-packs"
                  className="text-sm font-medium text-sky-700 hover:underline dark:text-sky-400"
                >
                  Back to concept packs
                </Link>
              }
            />
          </div>
        </div>
      );
    }
    return (
      <div className="px-8 py-6">
        <PageHeader title={name} />
        <div className="mt-6">
          <ErrorState
            title="Failed to load concept pack"
            message={packQ.error?.message}
            onRetry={() => packQ.refetch()}
          />
        </div>
      </div>
    );
  }

  const pack = packQ.data;
  if (!pack) return null;

  // Layer counts for the sidebar are not yet available from a stats endpoint.
  // Show 0..6 with a blank count; the active query reports the per-view total
  // in the right-hand pane header.
  const layerCounts = KNOWN_LAYERS.map((layer) => ({ layer, count: 0 }));

  return (
    <div className="px-8 py-6">
      <PageHeader
        title={pack.name}
        description={`v${pack.version}${pack.source_pack ? ` · forked from ${pack.source_pack}` : ""}`}
        actions={
          <>
            <Badge variant="muted">
              {pack.concept_count.toLocaleString()} concepts
            </Badge>
            {pack.simplex_count > 0 ? (
              <Badge variant="muted">{pack.simplex_count} simplexes</Badge>
            ) : null}
          </>
        }
      />

      <div className="mt-6 flex gap-6">
        <HierarchyBrowser
          layers={layerCounts}
          totalConcepts={pack.concept_count}
          selectedLayer={selectedLayer}
          onSelectLayer={setSelectedLayer}
        />

        <div className="flex-1 min-w-0">
          <Card>
            <CardHeader className="flex items-center justify-between gap-3">
              <CardTitle>
                {selectedLayer === null
                  ? "All concepts"
                  : `Layer ${selectedLayer}`}
              </CardTitle>
              <input
                type="search"
                placeholder="Search term, lemma, definition…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-64 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm placeholder:text-zinc-400 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100 dark:placeholder:text-zinc-500"
              />
            </CardHeader>
            <CardBody>
              {conceptsQ.isLoading ? (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Loading concepts…
                </p>
              ) : conceptsQ.isError ? (
                <ErrorState
                  title="Failed to load concepts"
                  message={conceptsQ.error?.message}
                  onRetry={() => conceptsQ.refetch()}
                />
              ) : !conceptsQ.data || conceptsQ.data.items.length === 0 ? (
                <EmptyState
                  title="No concepts match this view"
                  description={
                    query
                      ? `No concepts match "${query}"${selectedLayer !== null ? ` at layer ${selectedLayer}` : ""}.`
                      : "This layer has no concepts on disk."
                  }
                />
              ) : (
                <ConceptList
                  packName={name}
                  concepts={conceptsQ.data.items}
                  totalShown={conceptsQ.data.items.length}
                  total={conceptsQ.data.total}
                />
              )}
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ConceptList({
  packName,
  concepts,
  totalShown,
  total,
}: {
  packName: string;
  concepts: Concept[];
  totalShown: number;
  total: number;
}) {
  return (
    <>
      <p className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">
        Showing {totalShown.toLocaleString()} of {total.toLocaleString()}
      </p>
      <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
        {concepts.map((c) => (
          <li key={`${c.layer}:${c.term}`} className="py-2">
            <Link
              href={`/concept-packs/${encodeURIComponent(packName)}/concepts/${encodeURIComponent(c.term)}`}
              className="flex items-baseline justify-between gap-3 group"
            >
              <div className="min-w-0">
                <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100 group-hover:underline">
                  {c.term}
                </span>
                {c.definition ? (
                  <span className="ml-2 text-xs text-zinc-500 dark:text-zinc-400 line-clamp-1">
                    {c.definition}
                  </span>
                ) : null}
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {c.lens_pack_ids.length > 0 ? (
                  <Badge variant="info">
                    {c.lens_pack_ids.length} lens
                    {c.lens_pack_ids.length === 1 ? "" : "es"}
                  </Badge>
                ) : null}
                {c.safety_tags.map((tag) => (
                  <Badge key={tag} variant="warning">
                    {tag}
                  </Badge>
                ))}
                <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">
                  L{c.layer}
                </span>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </>
  );
}
