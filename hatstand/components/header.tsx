import Link from "next/link";
import { Suspense } from "react";
import { SearchBar } from "@/components/search-bar";

export function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/90 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/90">
      <div className="flex items-center gap-6 px-4 h-14">
        <Link
          href="/"
          className="flex items-center gap-2 shrink-0"
        >
          <span className="text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            HatStand
          </span>
          <span className="text-xs text-zinc-500 dark:text-zinc-400 hidden sm:inline">
            HatCat admin
          </span>
        </Link>
        <div className="flex-1 flex justify-center">
          {/* SearchBar uses useSearchParams which requires a Suspense boundary
              so static prerender (e.g. /_not-found) doesn't bail. */}
          <Suspense fallback={<div className="w-full max-w-xl h-9" />}>
            <SearchBar />
          </Suspense>
        </div>
      </div>
    </header>
  );
}
