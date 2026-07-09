// GET /api/admin/docs/file/{...relativePath}
// Reads a single markdown doc from the HatCatDev tree.
//
// Defense in depth: rejects any path containing ".." segments or absolute
// segments, and the resolved absolute path must remain within an allowed
// indexed location. Only .md files are served.

import path from "node:path";
import fs from "node:fs/promises";
import { getHatCatDevRoot } from "@/lib/server/hatcatdev";
import { parseFlatFrontmatter } from "@/lib/server/frontmatter";
import { ok, badRequest, notFound, fail } from "@/lib/server/api-helpers";
import type { Doc, DocFrontmatter, DocHeading } from "@/types";

const ALLOWED_PREFIXES = [
  "docs/",
  "melds/reference/",
  "melds/applied/",
  "concept_packs/",
  "src/lens_packs/",
];

const ALLOWED_ROOT_FILES = new Set([
  "README.md",
  "PROJECT_PLAN_PHASE_A.md",
  "PROJECT_PLAN_PHASE_B.md",
  "PROJECT_OVERVIEW.md",
  "QUICKSTART.md",
  "DEPLOYMENT.md",
  "TRAINING_QUICK_START.md",
]);

function isAllowed(relPath: string): boolean {
  if (ALLOWED_ROOT_FILES.has(relPath)) return true;
  return ALLOWED_PREFIXES.some((p) => relPath.startsWith(p));
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_~]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function extractHeadings(body: string): DocHeading[] {
  const lines = body.split(/\r?\n/);
  const headings: DocHeading[] = [];
  let inFence = false;
  for (const line of lines) {
    if (/^```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (!m) continue;
    const level = m[1].length;
    const text = m[2].trim();
    headings.push({ level, text, anchor: slugify(text) });
  }
  return headings;
}

function pickTitle(
  fm: DocFrontmatter,
  headings: DocHeading[],
  relPath: string,
): string {
  const fmTitle = fm["title"];
  if (typeof fmTitle === "string" && fmTitle.trim()) return fmTitle.trim();
  const firstH1 = headings.find((h) => h.level === 1);
  if (firstH1) return firstH1.text;
  const base = path.basename(relPath).replace(/\.md$/, "");
  return base;
}

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path: segments } = await ctx.params;
  if (!segments || segments.length === 0) {
    return badRequest("Missing path");
  }

  // Reject traversal and absolute fragments
  for (const seg of segments) {
    if (!seg || seg === "." || seg === ".." || seg.includes("\0")) {
      return badRequest("Invalid path segment", { segment: seg });
    }
  }

  const relPath = segments.join("/");
  if (!relPath.endsWith(".md")) {
    return badRequest("Only .md files are served", { path: relPath });
  }
  if (!isAllowed(relPath)) {
    return badRequest("Path is outside indexed sources", { path: relPath });
  }

  const root = getHatCatDevRoot();
  const absPath = path.join(root, relPath);
  // Final containment check
  const normalizedAbs = path.resolve(absPath);
  const normalizedRoot = path.resolve(root);
  if (
    normalizedAbs !== normalizedRoot &&
    !normalizedAbs.startsWith(normalizedRoot + path.sep)
  ) {
    return badRequest("Path escapes HatCatDev root");
  }

  let raw: string;
  let mtime: Date;
  try {
    [raw, mtime] = await Promise.all([
      fs.readFile(normalizedAbs, "utf-8"),
      fs.stat(normalizedAbs).then((s) => s.mtime),
    ]);
  } catch (err) {
    const e = err as NodeJS.ErrnoException;
    if (e.code === "ENOENT") return notFound(`Doc not found: ${relPath}`);
    return fail("docs_file_failed", e.message ?? String(err), 500);
  }

  const { fm, rest } = parseFlatFrontmatter(raw);
  const fmAsDoc: DocFrontmatter = fm;
  const headings = extractHeadings(rest);
  const folder = path.posix.dirname(relPath);

  const doc: Doc = {
    path: relPath,
    title: pickTitle(fmAsDoc, headings, relPath),
    folder: folder === "." ? "" : folder,
    frontmatter: fmAsDoc,
    body: rest,
    headings,
    inbound_links: [],
    outbound_links: [],
    updated_at: mtime.toISOString(),
  };

  return ok<Doc>(doc, "hatcatdev:fs");
}
