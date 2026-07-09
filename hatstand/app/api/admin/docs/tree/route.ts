// GET /api/admin/docs/tree
// Walks the indexed HatCatDev markdown sources and returns a single merged tree.
//
// Indexed sources (paths relative to HatCatDev root):
//   - docs/**/*.md
//   - melds/reference/*.md, melds/applied/*.md (when present)
//   - concept_packs/*/README.md
//   - src/lens_packs/*/README.md
//   - top-level: README.md, PROJECT_PLAN_PHASE_A.md, PROJECT_PLAN_PHASE_B.md,
//     PROJECT_OVERVIEW.md, QUICKSTART.md, DEPLOYMENT.md, TRAINING_QUICK_START.md
//
// The tree is rooted at a synthetic "HatCatDev" node and presents folders for
// docs/, melds/, concept_packs/ and src/lens_packs/, plus top-level .md files
// as direct leaves.

import path from "node:path";
import fs from "node:fs/promises";
import { getHatCatDevRoot } from "@/lib/server/hatcatdev";
import { ok, fail } from "@/lib/server/api-helpers";
import type { DocTreeNode } from "@/types";

const ROOT_MD_FILES = [
  "README.md",
  "PROJECT_PLAN_PHASE_A.md",
  "PROJECT_PLAN_PHASE_B.md",
  "PROJECT_OVERVIEW.md",
  "QUICKSTART.md",
  "DEPLOYMENT.md",
  "TRAINING_QUICK_START.md",
];

async function walkMarkdown(absDir: string, relDir: string): Promise<DocTreeNode[]> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(absDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const children: DocTreeNode[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    const absChild = path.join(absDir, entry.name);
    const relChild = relDir ? `${relDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      const sub = await walkMarkdown(absChild, relChild);
      if (sub.length > 0) {
        children.push({
          name: entry.name,
          path: relChild,
          is_directory: true,
          children: sub,
        });
      }
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      children.push({
        name: entry.name.replace(/\.md$/, ""),
        path: relChild,
        is_directory: false,
      });
    }
  }
  // Folders before files, then alphabetical
  children.sort((a, b) => {
    if (a.is_directory !== b.is_directory) return a.is_directory ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  return children;
}

async function readmesUnder(
  absParent: string,
  relParent: string,
): Promise<DocTreeNode[]> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(absParent, { withFileTypes: true });
  } catch {
    return [];
  }
  const children: DocTreeNode[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    const readmeRel = `${relParent}/${entry.name}/README.md`;
    const readmeAbs = path.join(absParent, entry.name, "README.md");
    try {
      await fs.access(readmeAbs);
    } catch {
      continue;
    }
    children.push({
      name: entry.name,
      path: readmeRel,
      is_directory: false,
    });
  }
  children.sort((a, b) => a.name.localeCompare(b.name));
  return children;
}

async function buildTree(root: string): Promise<DocTreeNode> {
  const top: DocTreeNode[] = [];

  // docs/
  const docsChildren = await walkMarkdown(path.join(root, "docs"), "docs");
  if (docsChildren.length > 0) {
    top.push({
      name: "docs",
      path: "docs",
      is_directory: true,
      children: docsChildren,
    });
  }

  // melds/ — only reference/ and applied/ subtrees, only their .md files
  const meldChildren: DocTreeNode[] = [];
  for (const sub of ["reference", "applied"]) {
    const subAbs = path.join(root, "melds", sub);
    const subChildren = await walkMarkdown(subAbs, `melds/${sub}`);
    if (subChildren.length > 0) {
      meldChildren.push({
        name: sub,
        path: `melds/${sub}`,
        is_directory: true,
        children: subChildren,
      });
    }
  }
  if (meldChildren.length > 0) {
    top.push({
      name: "melds",
      path: "melds",
      is_directory: true,
      children: meldChildren,
    });
  }

  // concept_packs/*/README.md
  const cpChildren = await readmesUnder(
    path.join(root, "concept_packs"),
    "concept_packs",
  );
  if (cpChildren.length > 0) {
    top.push({
      name: "concept_packs",
      path: "concept_packs",
      is_directory: true,
      children: cpChildren,
    });
  }

  // src/lens_packs/*/README.md
  const lpChildren = await readmesUnder(
    path.join(root, "src", "lens_packs"),
    "src/lens_packs",
  );
  if (lpChildren.length > 0) {
    top.push({
      name: "lens_packs",
      path: "src/lens_packs",
      is_directory: true,
      children: lpChildren,
    });
  }

  // Top-level root .md files
  for (const name of ROOT_MD_FILES) {
    try {
      await fs.access(path.join(root, name));
      top.push({
        name: name.replace(/\.md$/, ""),
        path: name,
        is_directory: false,
      });
    } catch {
      // missing → skip
    }
  }

  return {
    name: "HatCatDev",
    path: "",
    is_directory: true,
    children: top,
  };
}

export async function GET() {
  try {
    const root = getHatCatDevRoot();
    const tree = await buildTree(root);
    return ok<DocTreeNode>(tree, "hatcatdev:fs");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return fail("docs_tree_failed", message, 500);
  }
}
