import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Pin workspace root so Next doesn't infer the junk root package-lock
  // sitting in HatCatDev/.
  turbopack: {
    root: path.resolve(__dirname),
  },

  // Enable for portable / Docker / sovereign deployments. Produces a
  // self-contained server at .next/standalone/server.js after `next build`,
  // independent of node_modules. Default `next start` flow works without this.
  // output: "standalone",
};

export default nextConfig;
