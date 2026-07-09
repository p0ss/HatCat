"use client";

import Link from "next/link";
import { use } from "react";
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
import { useMeld } from "@/lib/hooks/use-melds";
import type { Meld, ProtectionLevel } from "@/types";

const PROTECTION_VARIANT: Record<
  ProtectionLevel,
  "muted" | "info" | "success" | "warning" | "error"
> = {
  open: "muted",
  guarded: "warning",
  sealed: "error",
};

export default function MeldDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: rawId } = use(params);
  const id = decodeURIComponent(rawId);
  const { data, isLoading, isError, error, refetch } = useMeld(id);

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
              title="Meld not found"
              description={`No meld file found for '${id}' under HatCatDev/melds/{pending,reference,applied}/.`}
              action={
                <Link
                  href="/melds"
                  className="text-sm font-medium text-sky-700 hover:underline dark:text-sky-400"
                >
                  Back to melds
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
            title="Failed to load meld"
            message={error?.message}
            onRetry={() => refetch()}
          />
        </div>
      </div>
    );
  }

  if (!data) return null;
  return <MeldDetail meld={data} />;
}

function MeldDetail({ meld }: { meld: Meld }) {
  const sectionLink =
    "block text-xs font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100";

  return (
    <div className="px-8 py-6">
      <PageHeader
        title={meld.id}
        description={`→ ${meld.target_pack} · ${meld.source}`}
        actions={
          <>
            <MeldStateBadge state={meld.state} />
            <Badge variant={PROTECTION_VARIANT[meld.protection_level]}>
              {meld.protection_level}
            </Badge>
          </>
        }
      />

      <nav className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        <a href="#overview" className={sectionLink}>Overview</a>
        <a href="#candidates" className={sectionLink}>Candidates</a>
        <a href="#structural-ops" className={sectionLink}>Structural ops</a>
        <a href="#impact" className={sectionLink}>Impact</a>
        <a href="#evidence" className={sectionLink}>Evidence</a>
        <a href="#reviews" className={sectionLink}>Reviews</a>
        <a href="#diff" className={sectionLink}>Diff</a>
      </nav>

      <div className="mt-6 grid gap-4">
        <Card id="overview">
          <CardHeader>
            <CardTitle>Overview</CardTitle>
          </CardHeader>
          <CardBody>
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <Field label="Source">{meld.source}</Field>
              <Field label="Target pack">
                <span className="font-mono text-xs break-all">{meld.target_pack}</span>
              </Field>
              <Field label="Protection level">
                <Badge variant={PROTECTION_VARIANT[meld.protection_level]}>
                  {meld.protection_level}
                </Badge>
              </Field>
              <Field label="State"><MeldStateBadge state={meld.state} /></Field>
              <Field label="Created at">
                <span className="text-xs">{meld.created_at || "—"}</span>
              </Field>
              <Field label="Updated at">
                <span className="text-xs">{meld.updated_at || "—"}</span>
              </Field>
            </dl>
          </CardBody>
        </Card>

        <Card id="candidates">
          <CardHeader>
            <CardTitle>Candidates ({meld.candidates.length})</CardTitle>
          </CardHeader>
          <CardBody>
            {meld.candidates.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No candidates declared.
              </p>
            ) : (
              <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {meld.candidates.map((c, i) => (
                  <li key={`${c.term ?? "candidate"}-${i}`} className="py-2 first:pt-0 last:pb-0">
                    <div className="flex items-baseline gap-2">
                      <Badge variant={c.kind === "concept" ? "info" : "muted"}>
                        {c.kind}
                      </Badge>
                      <span className="font-medium text-sm text-zinc-900 dark:text-zinc-100">
                        {c.term ?? "—"}
                      </span>
                      {c.parent ? (
                        <span className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
                          parent: <span className="font-mono">{c.parent}</span>
                        </span>
                      ) : null}
                    </div>
                    {c.children && c.children.length > 0 ? (
                      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                        children: {c.children.join(", ")}
                      </p>
                    ) : null}
                    {c.rationale ? (
                      <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
                        {c.rationale}
                      </p>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card id="structural-ops">
          <CardHeader>
            <CardTitle>Structural ops ({meld.structural_ops.length})</CardTitle>
          </CardHeader>
          <CardBody>
            {meld.structural_ops.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No structural operations declared.
              </p>
            ) : (
              <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {meld.structural_ops.map((op, i) => (
                  <li key={i} className="py-2 first:pt-0 last:pb-0">
                    <div className="flex items-baseline gap-2">
                      <Badge variant="warning">{op.op}</Badge>
                      <span className="font-mono text-xs text-zinc-700 dark:text-zinc-300">
                        {op.target}
                      </span>
                    </div>
                    {op.rationale ? (
                      <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
                        {op.rationale}
                      </p>
                    ) : null}
                    {op.details && Object.keys(op.details).length > 0 ? (
                      <pre className="mt-1 text-[11px] bg-zinc-50 dark:bg-zinc-950 rounded p-2 overflow-x-auto">
{JSON.stringify(op.details, null, 2)}
                      </pre>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </CardBody>
        </Card>

        <Card id="impact">
          <CardHeader>
            <CardTitle>Impact</CardTitle>
          </CardHeader>
          <CardBody>
            {!meld.impact ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No impact analysis on file.
              </p>
            ) : (
              <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                <Field label="Predicted version bump">
                  <Badge variant="info">{meld.impact.predicted_version_bump}</Badge>
                </Field>
                <Field label="Retraining concepts">
                  <span className="font-medium">{meld.impact.retraining_concepts.length}</span>
                </Field>
                <Field label="Deletion list">
                  <span className="font-medium">{meld.impact.deletion_list.length}</span>
                </Field>
                {meld.impact.retraining_concepts.length > 0 ? (
                  <div className="sm:col-span-3">
                    <dt className="text-xs text-zinc-500 dark:text-zinc-400">
                      Concepts to retrain
                    </dt>
                    <dd className="mt-1 text-xs font-mono text-zinc-700 dark:text-zinc-300 break-all">
                      {meld.impact.retraining_concepts.join(", ")}
                    </dd>
                  </div>
                ) : null}
                {meld.impact.deletion_list.length > 0 ? (
                  <div className="sm:col-span-3">
                    <dt className="text-xs text-zinc-500 dark:text-zinc-400">Deletions</dt>
                    <dd className="mt-1 text-xs font-mono text-zinc-700 dark:text-zinc-300 break-all">
                      {meld.impact.deletion_list.join(", ")}
                    </dd>
                  </div>
                ) : null}
              </dl>
            )}
          </CardBody>
        </Card>

        <Card id="evidence">
          <CardHeader>
            <CardTitle>Evidence</CardTitle>
          </CardHeader>
          <CardBody>
            {!meld.evidence ||
            ((!meld.evidence.exemplar_turns || meld.evidence.exemplar_turns.length === 0) &&
              (!meld.evidence.co_firing || meld.evidence.co_firing.length === 0)) ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No evidence on file.
              </p>
            ) : (
              <div className="grid gap-4">
                {meld.evidence.exemplar_turns && meld.evidence.exemplar_turns.length > 0 ? (
                  <div>
                    <h4 className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400 mb-2">
                      Exemplar turns
                    </h4>
                    <ul className="space-y-2">
                      {meld.evidence.exemplar_turns.map((t) => (
                        <li
                          key={t.id}
                          className="rounded border border-zinc-200 dark:border-zinc-800 p-2 text-xs"
                        >
                          <div className="flex justify-between gap-2">
                            <span className="font-mono">{t.id}</span>
                            {typeof t.flag_density === "number" ? (
                              <Badge variant="warning">
                                flag {(t.flag_density * 100).toFixed(0)}%
                              </Badge>
                            ) : null}
                          </div>
                          <p className="mt-1 text-zinc-700 dark:text-zinc-300">{t.excerpt}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {meld.evidence.co_firing && meld.evidence.co_firing.length > 0 ? (
                  <div>
                    <h4 className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400 mb-2">
                      Co-firing
                    </h4>
                    <ul className="space-y-1 text-xs">
                      {meld.evidence.co_firing.map((cf, i) => (
                        <li key={i} className="flex justify-between">
                          <span className="font-mono">{cf.concepts.join(" + ")}</span>
                          <span className="font-medium">{(cf.co_rate * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}
          </CardBody>
        </Card>

        <Card id="reviews">
          <CardHeader>
            <CardTitle>Reviews ({meld.reviews.length})</CardTitle>
          </CardHeader>
          <CardBody>
            {meld.reviews.length === 0 ? (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                No reviews recorded.
              </p>
            ) : (
              <ol className="space-y-3">
                {meld.reviews.map((r, i) => (
                  <li
                    key={`${r.reviewer}-${r.decided_at}-${i}`}
                    className="border-l-2 border-zinc-200 dark:border-zinc-800 pl-3"
                  >
                    <div className="flex items-baseline gap-2">
                      <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {r.reviewer}
                      </span>
                      <Badge
                        variant={
                          r.decision === "approve"
                            ? "success"
                            : r.decision === "reject"
                              ? "error"
                              : r.decision === "escalate"
                                ? "warning"
                                : "info"
                        }
                      >
                        {r.decision}
                      </Badge>
                      <span className="text-xs text-zinc-500 dark:text-zinc-400">
                        {r.decided_at}
                      </span>
                    </div>
                    {r.comment ? (
                      <p className="mt-1 text-xs text-zinc-700 dark:text-zinc-300">{r.comment}</p>
                    ) : null}
                  </li>
                ))}
              </ol>
            )}
          </CardBody>
        </Card>

        <Card id="diff">
          <CardHeader>
            <CardTitle>Diff</CardTitle>
          </CardHeader>
          <CardBody>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Conceptual diff against the target pack will appear here once the
              diff service is wired up.
            </p>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-zinc-900 dark:text-zinc-100">
        {children}
      </dd>
    </div>
  );
}
