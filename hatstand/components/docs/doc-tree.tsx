"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { DocTreeNode } from "@/types";

type DocTreeProps = {
  root: DocTreeNode;
  activePath?: string | null;
};

export function DocTree({ root, activePath }: DocTreeProps) {
  // Auto-expand any folder that contains the active path.
  const initiallyOpen = useMemo(
    () => collectAncestorPaths(root, activePath ?? null),
    [root, activePath],
  );

  return (
    <ul className="text-sm">
      {(root.children ?? []).map((child) => (
        <TreeNode
          key={child.path}
          node={child}
          activePath={activePath ?? null}
          initiallyOpen={initiallyOpen}
          depth={0}
        />
      ))}
    </ul>
  );
}

function TreeNode({
  node,
  activePath,
  initiallyOpen,
  depth,
}: {
  node: DocTreeNode;
  activePath: string | null;
  initiallyOpen: Set<string>;
  depth: number;
}) {
  const [open, setOpen] = useState(
    initiallyOpen.has(node.path) || depth === 0,
  );
  const indent = { paddingLeft: `${depth * 12 + 8}px` };

  if (node.is_directory) {
    return (
      <li>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          style={indent}
          className="flex w-full items-center gap-1.5 py-1 pr-2 text-left text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900"
        >
          <span className="inline-block w-3 text-zinc-500 dark:text-zinc-500">
            {open ? "▾" : "▸"}
          </span>
          <span className="truncate font-medium">{node.name}</span>
        </button>
        {open && node.children && node.children.length > 0 ? (
          <ul>
            {node.children.map((child) => (
              <TreeNode
                key={child.path}
                node={child}
                activePath={activePath}
                initiallyOpen={initiallyOpen}
                depth={depth + 1}
              />
            ))}
          </ul>
        ) : null}
      </li>
    );
  }

  const isActive = activePath === node.path;
  const href = `/docs/${node.path}`;
  return (
    <li>
      <Link
        href={href}
        style={indent}
        className={`flex items-center gap-1.5 py-1 pr-2 ${
          isActive
            ? "bg-zinc-200 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50"
            : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
        }`}
      >
        <span className="inline-block w-3 text-zinc-400 dark:text-zinc-600">·</span>
        <span className="truncate">{node.name}</span>
      </Link>
    </li>
  );
}

// Collect every directory path that is an ancestor of activePath.
function collectAncestorPaths(
  root: DocTreeNode,
  activePath: string | null,
): Set<string> {
  const ancestors = new Set<string>();
  if (!activePath) return ancestors;
  const found = walk(root, activePath, ancestors);
  if (!found) ancestors.clear();
  return ancestors;
}

function walk(
  node: DocTreeNode,
  target: string,
  ancestors: Set<string>,
): boolean {
  if (!node.is_directory) {
    return node.path === target;
  }
  for (const child of node.children ?? []) {
    if (walk(child, target, ancestors)) {
      ancestors.add(node.path);
      return true;
    }
  }
  return false;
}
