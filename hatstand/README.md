# HatStand

Admin UI for the HatCat lifecycle: model inventory, concept pack browsing, meld pipeline review, lens pack metadata, training/calibration run orchestration, and unified search across all resources.

Standalone Next.js 16 app that talks to a local HatCatDev backend. **Designed to be locally hostable**; vendor cloud platforms are not assumed and not required.

Scope and architecture: see [`../docs/planning/admin_ui_scope.md`](../docs/planning/admin_ui_scope.md).

## Local development

```bash
npm install
npm run dev
```

Visit http://localhost:3000.

The dev server expects HatCatDev to be reachable at `NEXT_PUBLIC_HATCATDEV_URL` (defaults to `http://localhost:8000`). Configure via `.env.local`:

```env
NEXT_PUBLIC_HATCATDEV_URL=http://localhost:8000
HATSTAND_ADMIN_TOKEN=replace-me
```

## Production hosting (self-hosted)

HatStand is built to run on operator-controlled infrastructure alongside HatCatDev. Two supported shapes:

### Node server (`next start`)

Long-running Node process, simplest setup:

```bash
npm run build
npm run start          # serves on PORT (default 3000)
```

Suitable when HatStand runs on the same host as HatCatDev (shared filesystem access, localhost API calls, single-operator deployments).

### Standalone bundle (Docker / portable)

For container deployments or moving the build artifact between machines, enable `output: 'standalone'` in `next.config.ts` (the line is present and commented). After build, the entire runnable app is at `.next/standalone/`:

```bash
npm run build
node .next/standalone/server.js
```

This produces a self-contained server with only the dependencies actually used at runtime. Pair with a multi-stage Dockerfile to ship a minimal image. See `node_modules/next/dist/docs/01-app/01-getting-started/17-deploying.md` for current Next.js 16 self-hosting guidance.

## Why not Vercel-by-default

HatCat is designed for multi-sovereign deployment with operator-controlled data flows. The admin UI must be co-located with HatCatDev and its data (lens packs, model caches, audit logs), not a third-party SaaS. Vercel works as a Next.js adapter but is not a default and is not in scope for the primary deployment story.

## Project structure

```
hatstand/
├── app/                    # Next.js 16 app router (pages + layouts + route handlers)
├── components/             # Shared UI components
├── lib/                    # API client, query helpers, utilities
├── types/                  # Shared TypeScript types (search contract, domain entities, API envelope)
├── public/                 # Static assets
├── next.config.ts          # Workspace root pinned, standalone output documented
└── package.json
```

## Phase status

Phase 0 of the parallelisation plan in [`../docs/planning/admin_ui_scope.md`](../docs/planning/admin_ui_scope.md). Vertical slices (P1.A Models, P1.B Lens Packs, etc.) build on top once Phase 0 closes.
