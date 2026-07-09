"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import type { Doc } from "@/types";

// Inter-doc link resolution. Markdown like [foo](../approach/bar.md) needs to
// resolve to /docs/<resolved-relative-path>. We do this in the link transformer.
function resolveDocHref(currentPath: string, href: string): string {
  if (!href) return href;
  // External / anchor / absolute → leave alone
  if (
    href.startsWith("http://") ||
    href.startsWith("https://") ||
    href.startsWith("mailto:") ||
    href.startsWith("#") ||
    href.startsWith("/")
  ) {
    return href;
  }
  // Resolve as a path relative to the current doc's directory
  const dir = currentPath.split("/").slice(0, -1).join("/");
  const segments = (dir ? `${dir}/${href}` : href).split("/");
  const resolved: string[] = [];
  for (const seg of segments) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      resolved.pop();
      continue;
    }
    resolved.push(seg);
  }
  const joined = resolved.join("/");
  // Only rewrite for .md targets — leave others (image refs, etc.) alone
  if (joined.endsWith(".md")) return `/docs/${joined}`;
  return href;
}

export function DocRenderer({ doc }: { doc: Doc }) {
  return (
    <article className="min-w-0">
      <header className="mb-6 border-b border-zinc-200 pb-4 dark:border-zinc-800">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          {doc.title}
        </h1>
        <p className="mt-1 font-mono text-xs text-zinc-500 dark:text-zinc-500">
          {doc.path}
        </p>
      </header>
      <div className="prose-hatcat">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[
            rehypeSlug,
            [
              rehypeAutolinkHeadings,
              {
                behavior: "wrap",
                properties: { className: "heading-anchor" },
              },
            ],
            rehypeHighlight,
          ]}
          components={{
            a({ href, children, ...rest }) {
              const resolved = resolveDocHref(doc.path, href ?? "");
              const isInternal = resolved?.startsWith("/docs/");
              return (
                <a
                  href={resolved}
                  {...(isInternal ? {} : { target: "_blank", rel: "noreferrer noopener" })}
                  {...rest}
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {doc.body}
        </ReactMarkdown>
      </div>
    </article>
  );
}
