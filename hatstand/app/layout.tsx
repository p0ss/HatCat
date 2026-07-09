import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";
import { Header } from "@/components/header";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "HatStand",
  description: "HatCat admin UI — lens pack lifecycle, melds, runs, search.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        <Providers>
          <div className="flex min-h-screen flex-col">
            <Header />
            <div className="flex flex-1 min-h-0">
              <Nav />
              <main className="flex-1 min-w-0">{children}</main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
