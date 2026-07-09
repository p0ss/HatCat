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
import { useConcept } from "@/lib/hooks/use-concept-packs";
import type { Concept } from "@/types";

export default function ConceptDetailPage({
  params,
}: {
  params: Promise<{ name: string; term: string }>;
}) {
  const { name: rawName, term: rawTerm } = use(params);
  const name = decodeURIComponent(rawName);
  const term = decodeURIComponent(rawTerm);
  const { data, isLoading, isError, error, refetch } = useConcept(name, term);

  if (isLoading) {
    return (
      <div className="px-8 py-6">
        <PageHeader title={term} description="Loading…" />
      </div>
    );
  }

  if (isError) {
    const isNotFound =
      (error as { code?: string } | undefined)?.code === "not_found";
    if (isNotFound) {
      return (
        <div className="px-8 py-6">
          <PageHeader title={term} />
          <div className="mt-6">
            <EmptyState
              title="Concept not found"
              description={`No concept '${term}' in pack '${name}'.`}
              action={
                <Link
                  href={`/concept-packs/${encodeURIComponent(name)}`}
                  className="text-sm font-medium text-sky-700 hover:underline dark:text-sky-400"
                >
                  Back to {name}
                </Link>
              }
            />
          </div>
        </div>
      );
    }
    return (
      <div className="px-8 py-6">
        <PageHeader title={term} />
        <div className="mt-6">
          <ErrorState
            title="Failed to load concept"
            message={error?.message}
            onRetry={() => refetch()}
          />
        </div>
      </div>
    );
  }

  if (!data) return null;
  return <ConceptDetail packName={name} concept={data} />;
}

function ConceptDetail({
  packName,
  concept,
}: {
  packName: string;
  concept: Concept;
}) {
  const subtitle = [
    concept.sumo_term && concept.sumo_term !== concept.term
      ? `sumo: ${concept.sumo_term}`
      : null,
    `layer ${concept.layer}`,
    concept.domain ? `domain: ${concept.domain}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="px-8 py-6">
      <div className="mb-2">
        <Link
          href={`/concept-packs/${encodeURIComponent(packName)}`}
          className="text-xs text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          ← {packName}
        </Link>
      </div>
      <PageHeader
        title={concept.term}
        description={subtitle}
        actions={
          <>
            <Badge variant="muted">L{concept.layer}</Badge>
            {concept.safety_tags.map((tag) => (
              <Badge key={tag} variant="warning">
                {tag}
              </Badge>
            ))}
          </>
        }
      />

      <div className="mt-6 grid gap-4">
        <Card id="identity">
          <CardHeader>
            <CardTitle>Identity</CardTitle>
          </CardHeader>
          <CardBody>
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <KV label="Term" value={concept.term} mono />
              <KV
                label="SUMO term"
                value={concept.sumo_term ?? "—"}
                mono={!!concept.sumo_term}
              />
              <KV label="Layer" value={String(concept.layer)} />
              <KV label="Domain" value={concept.domain ?? "—"} />
              <div className="sm:col-span-2">
                <dt className="text-xs text-zinc-500 dark:text-zinc-400">Safety tags</dt>
                <dd className="mt-1">
                  {concept.safety_tags.length === 0 ? (
                    <span className="text-zinc-500 dark:text-zinc-400 text-sm">—</span>
                  ) : (
                    <ul className="flex flex-wrap gap-1.5">
                      {concept.safety_tags.map((tag) => (
                        <li key={tag}>
                          <Badge variant="warning">{tag}</Badge>
                        </li>
                      ))}
                    </ul>
                  )}
                </dd>
              </div>
            </dl>
          </CardBody>
        </Card>

        <Card id="definition">
          <CardHeader>
            <CardTitle>Definition</CardTitle>
          </CardHeader>
          <CardBody>
            {concept.definition ? (
              <p className="text-sm text-zinc-800 dark:text-zinc-200">
                {concept.definition}
              </p>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No definition recorded.
              </p>
            )}
          </CardBody>
        </Card>

        <Card id="lemmas">
          <CardHeader>
            <CardTitle>Lemmas</CardTitle>
          </CardHeader>
          <CardBody>
            {concept.lemmas.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No lemmas recorded.
              </p>
            ) : (
              <BadgeList items={concept.lemmas} />
            )}
            {concept.synsets.length > 0 ? (
              <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                {concept.synsets.length} WordNet synset
                {concept.synsets.length === 1 ? "" : "s"} attached (offsets in
                API; not surfaced here).
              </p>
            ) : null}
          </CardBody>
        </Card>

        <Card id="relations">
          <CardHeader>
            <CardTitle>Relations</CardTitle>
          </CardHeader>
          <CardBody>
            <dl className="grid gap-4 text-sm">
              <RelationGroup
                label="Parents"
                packName={packName}
                terms={concept.parent_ids}
              />
              <RelationGroup
                label="Children"
                packName={packName}
                terms={concept.children_ids}
              />
              <RelationGroup
                label="Siblings"
                packName={packName}
                terms={concept.sibling_ids}
              />
            </dl>
          </CardBody>
        </Card>

        <Card id="simplexes">
          <CardHeader>
            <CardTitle>Simplex bindings</CardTitle>
          </CardHeader>
          <CardBody>
            {concept.simplex_bindings.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No simplex bindings recorded.
              </p>
            ) : (
              <ul className="space-y-1.5 text-sm">
                {concept.simplex_bindings.map((b) => (
                  <li
                    key={`${b.simplex_id}:${b.pole}`}
                    className="flex items-center gap-2"
                  >
                    <span className="font-mono text-xs text-zinc-700 dark:text-zinc-300">
                      {b.simplex_id}
                    </span>
                    <Badge
                      variant={
                        b.pole === "positive"
                          ? "success"
                          : b.pole === "negative"
                            ? "error"
                            : "muted"
                      }
                    >
                      {b.pole}
                    </Badge>
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card id="lens-packs">
          <CardHeader>
            <CardTitle>Lens packs containing this concept</CardTitle>
          </CardHeader>
          <CardBody>
            {concept.lens_pack_ids.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No trained lens for this concept yet.
              </p>
            ) : (
              <ul className="space-y-1.5 text-sm">
                {concept.lens_pack_ids.map((id) => (
                  <li key={id}>
                    <Link
                      href={`/lens-packs/${encodeURIComponent(id)}`}
                      className="font-mono text-xs text-sky-700 hover:underline dark:text-sky-400"
                    >
                      {id}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function KV({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd
        className={`mt-0.5 ${mono ? "font-mono text-xs" : ""} text-zinc-900 dark:text-zinc-100 break-words`}
      >
        {value}
      </dd>
    </div>
  );
}

function BadgeList({ items, mono }: { items: string[]; mono?: boolean }) {
  return (
    <ul className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <li key={item}>
          <Badge variant="muted">
            <span className={mono ? "font-mono" : ""}>{item}</span>
          </Badge>
        </li>
      ))}
    </ul>
  );
}

function RelationGroup({
  label,
  packName,
  terms,
}: {
  label: string;
  packName: string;
  terms: string[];
}) {
  return (
    <div>
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className="mt-1">
        {terms.length === 0 ? (
          <span className="text-zinc-500 dark:text-zinc-400">—</span>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {terms.map((term) => (
              <li key={term}>
                <Link
                  href={`/concept-packs/${encodeURIComponent(packName)}/concepts/${encodeURIComponent(term)}`}
                  className="inline-flex items-center rounded-md bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
                >
                  {term}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </dd>
    </div>
  );
}
