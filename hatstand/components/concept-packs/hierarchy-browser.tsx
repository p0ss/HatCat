"use client";

// Pure presentational left-rail browser. Lists layers 0..6 with concept counts
// per layer. The selected layer is reported back via onSelectLayer; the parent
// is responsible for fetching that layer's concepts and rendering them.

type LayerCount = { layer: number; count: number };

type HierarchyBrowserProps = {
  layers: LayerCount[]; // sorted ascending by layer
  totalConcepts: number;
  selectedLayer: number | null; // null = "All layers"
  onSelectLayer: (layer: number | null) => void;
};

export function HierarchyBrowser({
  layers,
  totalConcepts,
  selectedLayer,
  onSelectLayer,
}: HierarchyBrowserProps) {
  return (
    <aside className="w-64 shrink-0">
      <div className="rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="px-3 py-2.5 border-b border-zinc-200 dark:border-zinc-800">
          <h3 className="text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Layers
          </h3>
        </div>
        <ul className="px-1.5 py-1.5 space-y-0.5">
          <li>
            <LayerButton
              label="All layers"
              count={totalConcepts}
              active={selectedLayer === null}
              onClick={() => onSelectLayer(null)}
            />
          </li>
          {layers.length === 0 ? (
            <li className="px-2 py-2 text-xs text-zinc-500 dark:text-zinc-400">
              No layer data on disk.
            </li>
          ) : (
            layers.map(({ layer, count }) => (
              <li key={layer}>
                <LayerButton
                  label={`Layer ${layer}`}
                  count={count}
                  active={selectedLayer === layer}
                  onClick={() => onSelectLayer(layer)}
                />
              </li>
            ))
          )}
        </ul>
      </div>
    </aside>
  );
}

function LayerButton({
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
      className={`w-full flex items-center justify-between px-2 py-1.5 rounded-md text-sm transition-colors ${
        active
          ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50"
          : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
      }`}
    >
      <span>{label}</span>
      <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">
        {count.toLocaleString()}
      </span>
    </button>
  );
}
