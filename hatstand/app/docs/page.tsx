"use client";

import { PageHeader } from "@/components/page-header";
import { DocTree } from "@/components/docs/doc-tree";
import { EmptyState, ErrorState } from "@/components/ui";
import { useDocTree } from "@/lib/hooks/use-docs";

export default function DocsIndexPage() {
  const tree = useDocTree();

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Docs"
        description="HatCatDev markdown documentation — folder tree, rendered content, cross-links."
      />
      <div className="mt-6 grid grid-cols-[18rem_minmax(0,1fr)] gap-6">
        <aside className="sticky top-6 max-h-[calc(100vh-6rem)] overflow-y-auto rounded-lg border border-zinc-200 bg-white py-2 dark:border-zinc-800 dark:bg-zinc-900">
          {tree.isLoading ? (
            <p className="px-3 py-2 text-xs text-zinc-500 dark:text-zinc-500">
              Loading tree…
            </p>
          ) : tree.isError ? (
            <div className="p-3">
              <ErrorState
                title="Failed to load docs tree"
                message={tree.error.message}
                onRetry={() => tree.refetch()}
              />
            </div>
          ) : tree.data ? (
            <DocTree root={tree.data} activePath={null} />
          ) : null}
        </aside>
        <section className="min-w-0">
          <EmptyState
            title="No doc selected"
            description="Pick a markdown file from the tree to render it here."
          />
        </section>
      </div>
    </div>
  );
}
