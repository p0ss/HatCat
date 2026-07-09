type PlaceholderPaneProps = {
  resource: string;
  next?: string;
};

export function PlaceholderPane({ resource, next }: PlaceholderPaneProps) {
  return (
    <div className="mt-6 rounded-lg border border-dashed border-zinc-300 bg-white p-8 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400">
      <p>
        <span className="font-medium text-zinc-900 dark:text-zinc-100">{resource}</span> page
        — Phase 0 placeholder.
      </p>
      {next ? <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">{next}</p> : null}
    </div>
  );
}
