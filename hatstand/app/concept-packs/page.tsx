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
import { useConceptPacks } from "@/lib/hooks/use-concept-packs";
import type { ConceptPackSummary } from "@/types";

export default function ConceptPacksPage() {
  const { data, isLoading, isError, error, refetch } = useConceptPacks();

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Concept Packs"
        description="Concept ontology packs — hierarchy, simplexes, applied melds."
      />
      <div className="mt-6">
        {isLoading ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">Loading…</p>
        ) : isError ? (
          <ErrorState
            title="Failed to load concept packs"
            message={error?.message}
            onRetry={() => refetch()}
          />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No concept packs found"
            description="Expected one or more pack directories at HatCatDev/concept_packs/."
          />
        ) : (
          <ul className="grid gap-3 grid-cols-1 lg:grid-cols-2">
            {data.items.map((pack) => (
              <li key={pack.name}>
                <ConceptPackCard pack={pack} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function ConceptPackCard({ pack }: { pack: ConceptPackSummary }) {
  return (
    <Link
      href={`/concept-packs/${encodeURIComponent(pack.name)}`}
      className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 rounded-lg"
    >
      <Card className="hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors">
        <CardHeader className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="truncate">{pack.name}</CardTitle>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400 truncate">
              v{pack.version}
              {pack.source_pack ? ` · forked from ${pack.source_pack}` : ""}
            </p>
          </div>
          <Badge variant="muted">
            {pack.concept_count.toLocaleString()} concepts
          </Badge>
        </CardHeader>
        <CardBody>
          <dl className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Layers</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100">
                {pack.layer_count}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Simplexes</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100">
                {pack.simplex_count}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Updated</dt>
              <dd className="mt-0.5 font-medium text-zinc-900 dark:text-zinc-100 truncate">
                {formatDate(pack.updated_at)}
              </dd>
            </div>
          </dl>
        </CardBody>
      </Card>
    </Link>
  );
}

function formatDate(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toISOString().slice(0, 10);
  } catch {
    return iso.slice(0, 10);
  }
}
