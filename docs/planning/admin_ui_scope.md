# HatStand — HatCat Admin UI Scope

Status: planning  
Created: 2026-05-09  
Updated: 2026-05-09 — named HatStand; search-first redesign + pons merge integrated

**HatStand** is a standalone Next.js application for managing HatCat lens pack lifecycle: model inventory, concept pack ontology browsing, meld pipeline review, lens pack metadata, training/calibration run orchestration, and unified search across all resources.

The name fits HatCat's hat metaphor (a stand where hats hang) and signals its role: HatStand stands in front of the autonomic loop holding the operator-facing levers (observation, configuration, governance) so the loop itself stays untouched.

> **Naming note**: do not conflate with the three chat-UI surfaces in the stack:
> - `HatChat` (camelcase) = LibreChat fork — **primary chat UI going forward** (best of the three tried)
> - `hatcat-ui` (lowercase) = OpenWebUI fork — kept as a substrate-universality testcase
> - `HatCatDev/src/ui/` basic UI — kept as a substrate-universality testcase
>
> HatStand is none of these — it is a fresh Next.js app, dev-side tooling that lives inside HatCatDev.

## Why standalone (not a chat-UI extension)

Both `hatcat-ui` (OpenWebUI fork) and `HatChat` (LibreChat fork) are chat UXs. Admin/management screens — forms, tables, run dashboards, file browsers — are structurally awkward inside either, and mixing audiences (chat users vs developer/operator admins) confuses both. A standalone Next.js app, with **HatCatDev (post-merge with pons) as the backend**, fits the shape better:

- Different audience: researchers/operators, not chat users
- Different patterns: forms, tables, monitors, version diffs — not message threads
- Independent versioning from LibreChat releases
- Can run on its own port with admin-tier auth
- Per-token observability (lens markdown, steering panel) stays in hatchat where the lens components already live; this UI is for **production lifecycle of packs and substrates**, not subject inspection

## Pons → HatCatDev merge

The admin UI is the natural forcing function for consolidating pons into HatCatDev. Once admin UI exists as another consumer of pons, having three runtime artifacts (HatCatDev / pons / admin UI) plus hatchat is more boundary than it's worth. Pons is mostly an API skin over HatCatDev's data and processes — folding it in consolidates the runtime side to one repo.

**Merge scope** (lands as P0.0 in the parallelisation plan below):

- Relocate `pons/src/pons/*` modules into `HatCatDev/src/pons/` (preserves the conceptual identity and recognisable module names — bridge / containment / harness / lifecycle / audit / contract)
- Move pons CLI, blueprints, schemas into HatCatDev under appropriate locations
- Update hatchat's API target from pons's URL to HatCatDev's
- The `pons/hatchat-components/` TS package moves permanently into hatchat (it's already mirrored there as `HatChat/client/src/components/Lens/`); the canonical copy becomes the hatchat one
- Pons repo is archived after the migration

After the merge, HatCatDev is the deployed API service in addition to a research/training repo. The admin UI's `/v1/admin/*` endpoints land in HatCatDev directly, never living in a separate pons.

## Search architecture (first-order)

**Search is foundational, not a Phase 2 feature.** Every list/discovery view in the admin UI is a scoped search query against a single shared index. This unifies filtering across resources, enables cross-resource discovery (one query surfaces lens-pack entries, concept-pack concepts, melds, and docs), and dramatically simplifies the per-slice work for parallelising agents.

### Index schema

A single index of indexable documents across all resource types:

```
document:
  id              # globally unique (e.g. "lens_pack:gemma-4-e4b-...:concept:Refrigerator")
  resource_type   # concept_pack | lens_pack | concept | lens | simplex | meld | doc | model | run
  title           # display label
  body            # full-text searchable content
  facets          # JSON: layer, pack_id, status, model, simplex_dim, state, etc.
  url             # admin UI route to navigate to this resource
  parent_ids[]    # for hierarchy/scope filtering (e.g. concept belongs to pack X)
  updated_at
```

### Single search endpoint

```
GET /v1/admin/search
  ?q=<query>                       # optional, empty for "list everything"
  &type=<resource_type>            # optional, scopes results
  &filter[<facet_key>]=<value>     # repeatable, multi-valued via [] syntax
  &sort=<field>:<asc|desc>
  &cursor=<pagination_token>

Response:
{
  items: [{ id, resource_type, title, body_excerpt, url, facets, score, ... }],
  facets: { layer: {0: 5, 1: 23, ...}, status: {...} },
  total: number,
  next_cursor?: string
}
```

This endpoint powers:

- The Models page list view (`type=model`)
- The Lens Packs list view (`type=lens_pack`, facets for status/calibration/substrate)
- A lens pack's concepts tab (`type=lens`, `parent_ids=lens_pack:foo`)
- A lens pack's simplexes tab (`type=simplex`, `parent_ids=lens_pack:foo`)
- The universal search bar (no `type` filter)
- The Meld queue (`type=meld`, `filter[state]=review`)
- Docs (`type=doc`, `filter[folder]=specification`)
- Concept Packs list (`type=concept_pack`)
- Concept Pack hierarchy (`type=concept`, `parent_ids=concept_pack:bar`)

Per-page filter UIs become **facet selection** on the same response shape. No new bespoke list endpoints per slice — different default queries against the universal search.

### Backend choice

**SQLite FTS5** for v1:
- Zero extra service to deploy alongside HatCatDev
- Fast for HatCat's scale (a few thousand concepts × few packs × few hundred docs × growing meld history)
- Built-in BM25 ranking
- Index file lives in HatCatDev's data dir; rebuilt from filesystem on startup or via state-change webhook

If scale outgrows SQLite: swap to **Tantivy** or **Meilisearch** behind the same `/v1/admin/search` contract. The endpoint shape is the abstraction.

### Indexers

Each resource type has an **indexer** that produces documents and writes them to the index. Indexer interface is part of the P0.6 contract; each P1 slice implements its slice's indexer (~50-100 LOC per slice).

- **Static resources** (concept packs, lens packs, docs): indexed on read or via filesystem watch (`fs.watch`) for live updates
- **Dynamic resources** (runs, melds): indexed on state change via the resource's lifecycle hooks
- **Composite resources** (lens pack concepts/simplexes): indexed when the parent pack is touched

### What's NOT search-driven

- **Detail endpoints** (`GET /v1/admin/lens-packs/{id}`) stay bespoke — search is for discovery; detail views need fully-loaded structured data
- **Run logs (SSE streams)** — streaming, not appropriate for an index
- **Calibration data dumps** — large structured data, direct fetch
- **Live runs state** — querying "what's running now" is a small live-data endpoint, not search

Search covers ~80% of list/discovery surfaces; the remainder is detail/streaming/live-state with their own targeted endpoints.

## Top-level navigation

- **Dashboard** — overview, recent activity, run status
- **Models** — local cache + hub-available models
- **Concept Packs** — pack list + hierarchical browser
- **Melds** — ASK TRACE pipeline as a job queue (Tender → Review → Authorise → Commit → Evaluate)
- **Lens Packs** — pack list + per-pack metadata
- **Runs** (training/calibration/eval) — active and recent jobs
- **Docs** — markdown documentation viewer with hierarchy and cross-links
- **Search** — universal search across all resources (also accessible via global search bar in header)
- **Settings** — backend connection, env detection, HF token, defaults

## Models

**List view**: scoped search query (`type=model`). Facets for cached/partial/hub-only and family. Reads `~/.cache/huggingface/hub/` and checks completeness via the model indexer.

**Detail view**: manifest, snapshots/revisions, disk path, lens packs targeting this substrate. Actions: pull from hub, delete cache, set as default substrate.

## Concept Packs

**List view**: scoped search query (`type=concept_pack`). Facets for source_pack and version. Reads `concept_packs/*/pack.json` via the concept pack indexer.

**Detail view**, tabbed:
- **Hierarchy** — collapsible tree across layers 0-6, each node clickable. Backed by scoped search (`type=concept`, `parent_ids=concept_pack:{name}`) for filtering and search-within-pack.
- **Simplexes** — scoped search (`type=simplex`, `parent_ids=concept_pack:{name}`)
- **Melds Applied** — scoped search (`type=meld`, `filter[applied_to]={name}`)
- **Stats** — concepts per layer, per-domain breakdown, safety_tag counts (aggregate from facet counts)

**Concept detail panel** (selected from hierarchy): SUMO term, definition, layer, synsets, lemmas, parent/sibling/children, simplex bindings, safety tags, linked lens packs (via cross-resource search).

## Melds (ASK TRACE pipeline)

**List view**: scoped search query (`type=meld`). Facets: state (`tender|review|authorise|commit|evaluate|rejected`), source (manual|be_discovery|cat|cross_be|external), target_pack, protection_level. Visual state badges with workflow position.

**Detail view**, tabbed:
- **Candidates** — proposed concepts/relationships
- **Structural ops** — deprecate/merge/split/move with rationale
- **Impact analysis** — which existing lenses retrain, deletion list, version bump prediction
- **Evidence** — for CAT-derived melds, exemplar turns + co-firing data
- **Protection assessment** — auto-computed protection level + triggers
- **Reviews** — comments/decisions per ASK reviewer
- **Diff** — ontology before/after if applied

**Actions** (gated by reviewer permissions): approve, reject, request changes, escalate. Timeline view showing state transitions.

## Lens Packs

**List view**: scoped search query (`type=lens_pack`). Facets for status (trained|calibrating|uncalibrated|validated), substrate (model), version. Reads `src/lens_packs/.registry.json` plus per-pack manifests via the lens pack indexer.

**Detail view**, tabbed:
- **Aggregate metrics** — avg test_f1 per layer, calibration summary, HATComplianceReport summary
- **Hierarchy** — scoped search `type=lens, parent_ids=lens_pack:{id}` with hierarchy tree presentation. Each leaf shows lens metadata.
- **Simplex inventory** — scoped search `type=simplex, parent_ids=lens_pack:{id}` with per-pole metrics
- **Calibration data** — over-firer list, per-concept ConceptCalibration stats (sortable; could be a faceted search view too)
- **Provenance** — training/simplex/calibration run history (cross-resource search by `parent_ids=lens_pack:{id}, type=run`)

**Per-lens detail panel** (selected from hierarchy): file path, training metrics, calibration stats, selected layers, simplex binding (if any).

## Runs

**Active runs**: live-data endpoint (not search) — cards per active run with type, substrate, target pack, started_at, current step, progress bar, elapsed, ETA, log tail (streaming, last 20 lines). Actions: kill, view full log.

**Recent runs**: scoped search (`type=run`). Facets: type (training|simplex|calibration|eval), substrate, status (running|succeeded|failed|killed). Click into any: full config, full log, output artifacts.

**Launch new run**: form with type-dependent fields, pre-fills with last-used or sane defaults, records full config to per-run JSON for reproducibility, validates env first (right venv? required cache? GPU?) before launching, spawns subprocess with logs streamed to UI.

## Docs

Three-column layout (collapsible to two on narrow screens):
- **Left**: collapsible folder tree of indexed markdown sources
- **Centre**: rendered markdown
- **Right**: in-doc TOC auto-generated from headings, with current-section highlight on scroll

**Search within docs**: scoped search (`type=doc`) — same infrastructure as everywhere else. Folder tree mirrors filesystem; folder selection becomes a `filter[folder]=...` facet.

**Indexed sources** (configured in the docs indexer):
- `HatCatDev/docs/**/*.md`
- `HatCatDev/README.md`, root-level `PROJECT_PLAN_*.md` and `PROJECT_OVERVIEW.md`
- `HatCatDev/melds/reference/*.md` and `HatCatDev/melds/applied/*.md`
- `HatCatDev/concept_packs/*/README.md`
- `HatCatDev/src/lens_packs/*/README.md`

**Features**:
- Folder tree with type-icon hints
- GitHub-flavored markdown (tables, code blocks, task lists, strikethrough)
- Mermaid diagram rendering
- Inter-doc link resolution: relative-path markdown links resolve to docs UI routes
- Anchor links to headings (inbound URL hash + outbound clickable copy-permalink)
- Full-text search via the universal search infrastructure
- Live reload during dev via `fs.watch`
- Code copy buttons
- Renders `.kif` (SUMO source) and `.json` (manifest examples)

**Cross-app linkability**: lens pack pages link into MAP_MELDING.md, ARCHITECTURE.md, etc. Melds in the pipeline link to relevant operation specs. Docs viewer can open in a side panel without leaving the current admin page.

## Settings

Backend connection (HatCatDev URL, auth token), active Python env detection (venv vs conda) with versions, HF cache location and size, default substrate, output directory roots per pack type.

## Backend endpoints needed

After the pons → HatCatDev merge, all admin endpoints live in HatCatDev under `/v1/admin/*`.

```
# Universal search (powers most list/discovery views)
GET    /v1/admin/search?q=&type=&filter[*]=&sort=&cursor=

# Detail endpoints (bespoke per resource type)
GET    /v1/admin/models/{id}
POST   /v1/admin/models/{id}/pull
DELETE /v1/admin/models/{id}/cache

GET    /v1/admin/concept-packs/{name}
GET    /v1/admin/concept-packs/{name}/concepts/{term}

GET    /v1/admin/lens-packs/{id}
GET    /v1/admin/lens-packs/{id}/concepts/{term}
GET    /v1/admin/lens-packs/{id}/calibration
GET    /v1/admin/lens-packs/{id}/over-firers
GET    /v1/admin/lens-packs/{id}/compliance-report

GET    /v1/admin/melds/{id}
GET    /v1/admin/melds/{id}/impact
GET    /v1/admin/melds/{id}/diff
POST   /v1/admin/melds                   # submit
POST   /v1/admin/melds/{id}/review
POST   /v1/admin/melds/{id}/approve
POST   /v1/admin/melds/{id}/reject
POST   /v1/admin/melds/{id}/execute      # commit phase

# Runs (subprocess management — bespoke)
GET    /v1/admin/runs/{id}
GET    /v1/admin/runs/{id}/log            # SSE stream
POST   /v1/admin/runs                     # launch
POST   /v1/admin/runs/{id}/kill
GET    /v1/admin/runs/active              # live-data, not search

# Docs
GET    /v1/admin/docs/tree                # folder structure for tree view
GET    /v1/admin/docs/file/{path}         # markdown content + frontmatter

# Settings/diagnostics
GET    /v1/admin/env
GET    /v1/admin/registry
GET    /v1/admin/health

# Index management (admin/debug)
POST   /v1/admin/search/reindex           # rebuild full index
POST   /v1/admin/search/reindex/{type}    # rebuild one resource type
```

Note: bespoke list endpoints per resource type are intentionally absent — those are scoped queries against `/v1/admin/search`. List endpoints would be redundant (and inconsistent in shape).

## Tech stack

- **Next.js 14+** (app router) — SSR for static data, client-side for live state
- **TypeScript** throughout
- **shadcn/ui** + **Tailwind CSS** for components
- **TanStack Query** for data fetching with cache invalidation
- **TanStack Table** for heavy tables (lens lists, calibration data)
- **React Hook Form + Zod** for run-launcher forms
- **react-d3-tree** or similar for hierarchy visualization
- **react-markdown** + `remark-gfm` for markdown rendering
- **rehype-highlight** for code syntax highlighting
- **mermaid** for diagram rendering
- **EventSource** / SSE for streaming log tails
- **SQLite FTS5** (server-side, in HatCatDev) for the search index
- **fs.watch** (server-side, dev-only) for live re-indexing of static files

## MVP split

**v0.1 — addresses immediate pain points** (search, runs, basic inspection):
1. Pons → HatCatDev merge complete (P0.0)
2. Search infrastructure with index, indexers for models/lens-packs/runs/docs (P0.6)
3. Models page (read-only)
4. Lens Packs list + basic detail (read-only) — searchable list, status, aggregate metrics
5. Runs page with launcher, monitor, log streaming, env validation
6. Settings showing env detection
7. Docs viewer with folder tree + markdown rendering + cross-links + search

**v0.2 — adds inspection depth**:
8. Lens pack detail with per-lens metadata, calibration stats
9. Concept pack browser with hierarchy tree (with concept indexer)
10. Provenance chains and based_on lineage
11. Cross-page linking (lens-pack → docs, etc.)

**v0.3 — meld pipeline**:
12. Meld queue with ASK TRACE state machine
13. Review/approve flow
14. Impact analysis preview
15. Diff viewer

**v0.4 — observability hooks** (when behavioural validation lands):
16. Behavioural validation run launcher
17. HATComplianceReport visualization
18. Cross-pack comparison

## Out of scope

- Per-token observability — stays in hatchat where the lens components live
- Steering controls — governed via ASK, not UI levers (no-operator-intervention principle)
- Subject inference / chat — hatchat's domain
- BE governance / treaty management — different surface, possibly later
- Editing concept pack content directly — should go through the Meld pipeline, not direct file edit

## Implementation notes

- Backend lives in HatCatDev (post pons-merge). New endpoints under `/v1/admin/`. Most discovery surfaces are search queries with appropriate `type` and `filter` parameters; only detail/streaming/state-change endpoints are bespoke.
- Authentication: admin-tier auth, distinct from end-user auth. Possibly leverage existing pons auth machinery (now in HatCatDev), or simple env-token for dev/single-operator deployments.
- The UI should be runnable as a local dev tool (single operator) and as a deployed service (multi-user team). Auth scales with deployment mode.
- Run subprocesses managed by HatCatDev (not the UI directly) — UI submits config and watches output. Subprocess survives UI page reload.
- Search index lives in HatCatDev's data dir as a SQLite database. Rebuilt from filesystem on startup. Indexers are registered as part of each P1 slice and called when relevant data changes (filesystem watch for static, lifecycle hooks for dynamic).
- Filter UIs are facet panels driven by the search response's `facets` field. Adding a new filter is a matter of adding a facet to the indexer's document — no new endpoint or UI plumbing required.

---

## Parallelisation plan

Structure: a small **Phase 0** seed lands the contracts + scaffolding once; **Phase 1** vertical slices (one resource per agent) build in parallel; **Phase 2** integrates cross-cutting features. Search-as-foundation simplifies Phase 1 considerably — every slice has the same shape.

### Phase 0 — Seed (sequential, single agent or coordinated pair)

These items must land before any vertical slice can start.

**P0.0 — Pons → HatCatDev merge**
- Relocate `pons/src/pons/*` to `HatCatDev/src/pons/`, update imports
- Move pons CLI, blueprints, schemas
- Update hatchat's API target URL
- `pons/hatchat-components/` becomes hatchat-canonical (drop the pons copy)
- Verify hatchat still works against new HatCatDev API surface
- Archive pons repo / mark as deprecated

**P0.1 — Repo scaffold**
- Create `hatcat-admin/` Next.js 14+ project with TypeScript, Tailwind, shadcn/ui initialised
- Top-level layout component with navigation skeleton (placeholders for all pages)
- Routing for all top-level pages (`/`, `/models`, `/concept-packs`, `/melds`, `/lens-packs`, `/runs`, `/docs`, `/search`, `/settings`)
- Tailwind theme + dark mode
- TanStack Query provider wired
- Basic env config (`NEXT_PUBLIC_HATCATDEV_URL`)

**P0.2 — Shared types**
- `types/api.ts` defining response shapes for every HatCatDev admin endpoint
- `types/domain.ts` for cross-cutting domain types (LensPack, ConceptPack, Meld, RunStatus, SearchDocument, SearchResponse, Facet, etc.)
- Shape of the search request/response (the universal contract)

**P0.3 — HatCatDev admin namespace + auth**
- Add `/v1/admin/*` routing under HatCatDev
- Admin-tier auth middleware (env-token-based for local/single-operator)
- Health check endpoint (`/v1/admin/health`)
- Trivial endpoint working end-to-end so frontend can verify wiring

**P0.4 — Shared frontend components**
- `<DataTable>` (TanStack Table wrapper with shadcn)
- `<DetailLayout>` (header + tabs + side panel)
- `<HierarchyTree>` (collapsible tree, generic node type)
- `<StatusBadge>`
- `<MarkdownRenderer>`
- `<SearchBar>` — global search input
- `<SearchResults>` — universal result list with cross-resource result types
- `<FacetPanel>` — sidebar facet filter component, driven by search response's facets

**P0.5 — API client**
- `lib/api.ts` typed fetcher functions
- Error handling, retry policy, auth header injection
- TanStack Query hook factories per endpoint
- `useSearch(query)` hook for the universal search surface

**P0.6 — Search infrastructure**
- SQLite FTS5 index initialisation in HatCatDev
- Document schema definition + storage layer
- Indexer interface (`Indexer` base class or protocol) — slices implement this
- `/v1/admin/search` endpoint with query parsing, faceting, BM25 ranking, pagination
- `/v1/admin/search/reindex[/{type}]` admin endpoint for rebuilds
- Filesystem watcher for static-resource indexers
- Universal search page (`/search`) using the shared components

After P0 lands, all P1 vertical slices can start in parallel. P0 is more substantial than the previous plan due to (a) the pons merge and (b) search infrastructure — estimate 3-4 days for one focused agent (or two coordinated agents splitting the merge from the search infra).

### Phase 1 — Vertical slices (parallel)

Each slice owns one resource end-to-end: indexer + detail endpoint + frontend page. The shape is now uniform across slices because the list view comes for free from search infrastructure:

| Slice | Indexer | Detail endpoint(s) | Frontend page | Depends on |
|-------|---------|--------------------|---------------|-----------:|
| **P1.A — Models** | `model_indexer` (HF cache scan) | `GET /admin/models/{id}` + pull/cache actions | `/models` list (search-scoped) + `/models/[id]` detail | P0 |
| **P1.B — Lens Packs** | `lens_pack_indexer` + `lens_indexer` + `simplex_indexer` (per-pack-content) | `GET /admin/lens-packs/{id}`, calibration, over-firers, compliance-report | `/lens-packs` list + `/lens-packs/[id]` detail with tabs | P0 |
| **P1.C — Concept Packs** | `concept_pack_indexer` + `concept_indexer` | `GET /admin/concept-packs/{name}`, `GET /admin/concept-packs/{name}/concepts/{term}` | `/concept-packs` list + `/concept-packs/[name]` detail with hierarchy tree | P0 |
| **P1.D — Runs** | `run_indexer` (state-hooked) | `GET /admin/runs/{id}`, `POST /admin/runs`, `POST /admin/runs/{id}/kill`, SSE log stream | `/runs` list + active monitor + launcher form | P0 + needs subprocess manager in HatCatDev |
| **P1.E — Docs** | `doc_indexer` (markdown content extraction with frontmatter parsing) | `GET /admin/docs/tree`, `GET /admin/docs/file/{path}` | `/docs` viewer with tree + markdown + scoped search | P0 |
| **P1.F — Settings/Dashboard** | (no indexer needed) | `GET /admin/env`, `GET /admin/registry` | `/` dashboard + `/settings` form | P0 |
| **P1.G — Melds** | `meld_indexer` (state-hooked) | All meld endpoints (10+) including review/approve/execute | `/melds` queue + `/melds/[id]` detail with review flow | P0 + meld state machine in HatCatDev |

**Per-slice work shape** (each P1 slice has the same pattern, ~3-5 days each):
1. Define the document shape for this resource type (1-2 hours; goes in P0.2 / contract addendum)
2. Implement the indexer: read source data, produce documents, write to index (50-100 LOC)
3. Implement detail endpoint(s) — only the bespoke parts, no list endpoint needed
4. Build the page using `<SearchResults>` for the list view and `<DetailLayout>` for detail
5. Define facets relevant to this resource (concept pack: layer, domain; meld: state, source, protection_level; etc.)

**Concurrency model**: P1.A, P1.B, P1.C, P1.E, P1.F can run fully in parallel — they're independent indexer + detail + page work. P1.D (Runs) and P1.G (Melds) are heavier on the backend side because they involve state management (subprocess lifecycle, ASK TRACE state machine), so they may take longer or need an extra backend-focused agent.

### Phase 2 — Integration (sequential after P1)

**P2.1 — Cross-page linking**: lens pack pages link into doc viewer; meld details link to MAP_MELDING.md sections; concept pack browser links to lens packs that target each concept. Touches all P1 slices' navigation.

**P2.2 — Real-time updates**: WebSocket or SSE channels for live updates across pages (run progress, registry changes, index updates). Coordinates with the per-slice fetcher hooks.

**P2.3 — Search ranking polish**: cross-resource result grouping, smarter ranking signals (boost recent, downrank archived, etc.), suggestion/autocomplete, query suggestions. (Universal search infrastructure already in v0.1; this is presentation polish.)

**P2.4 — Provenance graph**: visualisation of based_on lineage across packs and melds. Requires graph data assembled from multiple slice endpoints.

**P2.5 — Bulk actions**: select multiple packs/concepts/melds for batch operations.

### Cross-slice contracts

Codified in P0.2 and P0.4:

1. **Search response shape** is universal: `{ items, facets, total, next_cursor }`. Every list view consumes it.
2. **All detail endpoints** return `{ data: T, _meta: { fetched_at, source } }`
3. **All error responses** use `{ error: { code, message, details? } }`
4. **All long-running operations** return job-handle responses immediately + provide a polling/SSE endpoint
5. **Routes are kebab-case in URLs**, camelCase in code, with `[id]`/`[name]` slugs
6. **Component props for shared components** are typed once in P0.4 and frozen
7. **Indexer interface** is fixed in P0.6: `index(resource) → SearchDocument[]` — slices implement this consistently

### Coordination points

Locked in P0 to avoid Phase 2 retrofitting:

- **Auth model** — P0.3
- **Subprocess management** — daemon spawned by HatCatDev; coordinated between P1.D backend and frontend
- **Real-time channel** — WebSocket vs SSE vs polling; chosen in P0
- **Search backend choice** — SQLite FTS5 for v1, with pluggable interface so v2 can swap to Tantivy/Meilisearch — chosen in P0.6
- **Indexer lifecycle** — when do indexers run? On startup, on filesystem change, on state change? Defined in P0.6.

### Recommended agent assignments for v0.1 MVP

| Agent | Slice | Effort estimate |
|-------|-------|-----------------|
| Agent 0 | P0 seed (split into two if separating P0.0 merge from rest) | 3-4 days (or 2+2 split) |
| Agent 1 | P1.A Models | 2-3 days |
| Agent 2 | P1.B Lens Packs (read-only basic) | 3-4 days |
| Agent 3 | P1.D Runs (most substantive — needs subprocess manager) | 4-5 days |
| Agent 4 | P1.E Docs | 2-3 days |
| Agent 5 | P1.F Settings + Dashboard | 1-2 days |
| Agent 6 | P2.1 Cross-page linking + light integration after P1 lands | 1-2 days |

P1 work proceeds concurrently after P0 closes. Total wall-clock for MVP: ~1 week if 5 agents work in parallel after the seed, vs. ~3-4 weeks for one agent serially. The contracts in P0 — especially the search contract — are what make that speedup real.

### Splitting further (per-slice sub-agents)

Each P1 slice can be split into:
- **Backend sub-agent**: implements indexer + detail endpoint(s) + tests
- **Frontend sub-agent**: implements page + slice-specific components, uses mocked search responses from contract types until backend lands

Sub-agents coordinate only on the document shape and detail endpoint contracts (locked in P0.2). If contracts are right, they develop fully independently and integrate at the end.

### What NOT to parallelise

- **P0 work** — has to land coherently as a foundation (especially the merge and search infrastructure)
- **Auth model decisions** — one consistent answer across all slices
- **Search index schema** — extensions are fine after P0; breaking changes mid-Phase 1 break all slices
- **Shared component contracts** — frozen in P0
- **Indexer interface** — frozen in P0.6
- **Schema migrations or breaking type changes** — coordinated rollout required
