// Universal search contract — every list/discovery view in HatStand is a
// scoped query against /v1/admin/search. See docs/planning/admin_ui_scope.md
// "Search architecture (first-order)".

export type ResourceType =
  | "model"
  | "concept_pack"
  | "concept"
  | "lens_pack"
  | "lens"
  | "simplex"
  | "meld"
  | "run"
  | "doc";

export type FacetKey =
  // shared
  | "updated_at"
  // models
  | "family"
  | "cached"
  // concept packs / concepts
  | "layer"
  | "domain"
  | "safety_tag"
  // lens packs / lenses / simplexes
  | "substrate"
  | "status"
  | "calibration_status"
  | "simplex_dim"
  | "simplex_state"
  // melds
  | "state"
  | "source"
  | "protection_level"
  | "target_pack"
  | "applied_to"
  // runs
  | "run_type"
  | "run_status"
  // docs
  | "folder"
  // generic
  | "tag";

export type FacetValueCount = {
  value: string;
  count: number;
  label?: string;
};

export type FacetCounts = Partial<Record<FacetKey, FacetValueCount[]>>;

export type SearchDocument = {
  id: string;
  resource_type: ResourceType;
  title: string;
  body_excerpt?: string;
  url: string;
  facets: Partial<Record<FacetKey, string | string[]>>;
  parent_ids: string[];
  updated_at: string; // ISO 8601
  score?: number;
};

export type SortDirection = "asc" | "desc";

export type SearchSort = {
  field: string;
  direction: SortDirection;
};

export type SearchRequest = {
  q?: string;
  type?: ResourceType;
  filter?: Partial<Record<FacetKey, string | string[]>>;
  sort?: SearchSort;
  cursor?: string;
  limit?: number;
};

export type SearchResponse = {
  items: SearchDocument[];
  facets: FacetCounts;
  total: number;
  next_cursor?: string;
};
