// Domain types — one per resource. Detail-endpoint payloads.
// Search results use SearchDocument from ./search; these types describe
// the full bespoke record returned from GET /v1/admin/{resource}/{id}.

// ---------- Models (substrates) ----------

export type ModelStatus = "cached" | "partial" | "hub_only" | "error";

export type ModelSnapshot = {
  revision: string;
  size_bytes: number;
  files: number;
  is_default: boolean;
};

export type Model = {
  id: string; // e.g. "google/gemma-4-E4B"
  family: string; // e.g. "gemma-4"
  status: ModelStatus;
  disk_path?: string;
  size_bytes?: number;
  snapshots: ModelSnapshot[];
  default_snapshot?: string;
  lens_packs_targeting: string[]; // ids
  updated_at: string;
};

// ---------- Concept packs / concepts ----------

export type ConceptPackSummary = {
  name: string;
  version: string;
  source_pack?: string;
  concept_count: number;
  simplex_count: number;
  layer_count: number;
  created_at: string;
  updated_at: string;
};

export type SimplexBinding = {
  simplex_id: string;
  pole: "positive" | "neutral" | "negative";
};

export type Concept = {
  term: string; // lowercase, snake_case
  sumo_term?: string;
  definition?: string;
  layer: number; // 0..6
  synsets: string[];
  lemmas: string[];
  parent_ids: string[];
  sibling_ids: string[];
  children_ids: string[];
  simplex_bindings: SimplexBinding[];
  safety_tags: string[];
  domain?: string;
  lens_pack_ids: string[]; // packs that contain a trained lens for this concept
};

// ---------- Lens packs / lenses / simplexes ----------

export type LensPackStatus =
  | "trained"
  | "calibrating"
  | "uncalibrated"
  | "validated"
  | "error";

export type CalibrationStatus = "complete" | "partial" | "missing";

export type LensPackAggregateMetrics = {
  avg_test_f1_per_layer: Record<string, number>; // layer -> f1
  calibration_summary?: {
    // Number of (concept, layer) entries in calibration.json — note this
    // counts concept × layer combinations, NOT distinct concepts.
    entries_calibrated: number;
    // Number of distinct concepts represented across those entries.
    concepts_calibrated: number;
    // Number of concepts with trained lenses (per version_manifest).
    concepts_total: number;
    // Real count from calibration data, threshold = 0.3 cross_fire_rate.
    over_firers: number;
    // For runs that have a noise track, separate count.
    noise_over_firers?: number;
  };
  hat_compliance?: {
    locality?: number;
    transduction?: number;
    calibration?: number;
    efficiency?: number;
    control_authority?: number;
  };
};

export type LensPack = {
  id: string;
  substrate: string; // model id
  concept_pack: string; // pack name
  version: string;
  status: LensPackStatus;
  calibration_status: CalibrationStatus;
  aggregate_metrics: LensPackAggregateMetrics;
  registry_path?: string;
  created_at: string;
  updated_at: string;
  based_on?: string; // parent lens pack id
};

export type LensTrainingMetrics = {
  test_f1: number;
  test_precision?: number;
  test_recall?: number;
  test_accuracy?: number;
  selected_layer: number;
  trained_at: string;
};

// Per-(concept, layer) calibration record. Mirrors the on-disk
// calibration.json entries — fields differ slightly between calibration
// modes (gemma-3 v2 has gen_*, gemma-4 v1 has noise_*), so the mode-specific
// fields are optional.
export type ConceptCalibration = {
  concept: string;
  layer: number;
  self_mean: number;
  self_std: number;
  cross_mean: number;
  cross_std: number;
  cross_fire_count: number;
  cross_fire_rate: number;
  times_loaded: number;
  n_self_samples: number;
  n_cross_samples: number;
  // Generation-mode fields (older `merged` calibration runs)
  gen_mean?: number;
  gen_fire_count?: number;
  gen_fire_rate?: number;
  // Noise-mode fields (newer calibration runs)
  noise_mean?: number;
  noise_std?: number;
  noise_max?: number;
  noise_fire_count?: number;
  noise_fire_rate?: number;
};

export type CalibrationOverFirerSummary = {
  threshold: number;
  total_entries: number;
  over_firers: number;
  noise_over_firers?: number;
  per_layer: Array<{
    layer: number;
    total: number;
    over_firers: number;
    noise_over_firers?: number;
  }>;
};

// Per-cycle progression record extracted from the analysis + finetune file pair.
export type CalibrationCycle = {
  cycle: number;
  analysis_timestamp?: string;
  finetune_timestamp?: string;
  total_concepts: number;
  top_k?: number;
  avg_in_top_k_rate?: number;
  well_calibrated: number;
  under_firing: number;
  over_firing: number;
  finetune?: {
    total_lenses_processed: number;
    lenses_boosted: number;
    lenses_suppressed: number;
    avg_improvement: number;
  };
};

export type LensPackCalibration = {
  pack_id: string;
  mode?: string;
  has_noise_track: boolean;
  summary: CalibrationOverFirerSummary;
  cycles: CalibrationCycle[];
};

export type Lens = {
  pack_id: string;
  term: string;
  layer: number; // ontological layer of the concept
  selected_layer: number; // model layer the classifier was trained on
  file_path: string;
  training_metrics: LensTrainingMetrics;
  calibration?: ConceptCalibration;
  simplex_binding?: SimplexBinding;
};

export type SimplexState =
  | "active_positive"
  | "active_negative"
  | "active_blended"
  | "implicit_positive"
  | "implicit_negative"
  | "implicit_blended"
  | "neutral"
  | "not_relevant";

export type SimplexPole = {
  name: "positive" | "neutral" | "negative";
  // Short human label for the pole (e.g. "preference" / "indifference" / "aversion").
  // Comes from simplex/_definitions.json on disk.
  label?: string;
  // Full pole definition text from the same source.
  definition?: string;
  synsets: string[];
  test_f1?: number;
};

export type Simplex = {
  id: string;
  pack_id: string;
  dimension: string; // e.g. "happysad", "aspiration_social_mobility"
  poles: SimplexPole[];
  trained_at?: string;
  validation?: {
    in_distribution_f1?: number;
    cross_simplex_f1?: number;
    pole_disambiguation_f1?: number;
  };
};

// ---------- Melds ----------

export type MeldState =
  | "tender"
  | "review"
  | "authorise"
  | "commit"
  | "evaluate"
  | "rejected";

export type MeldSource =
  | "manual"
  | "be_discovery"
  | "cat"
  | "cross_be"
  | "external";

export type ProtectionLevel = "open" | "guarded" | "sealed";

export type MeldCandidate = {
  kind: "concept" | "relationship";
  term?: string;
  parent?: string;
  children?: string[];
  rationale?: string;
};

export type MeldStructuralOp = {
  op: "deprecate" | "merge" | "split" | "move";
  target: string;
  details: Record<string, unknown>;
  rationale: string;
};

export type MeldImpact = {
  retraining_concepts: string[];
  deletion_list: string[];
  predicted_version_bump: "major" | "minor" | "patch";
};

export type MeldEvidence = {
  exemplar_turns?: Array<{ id: string; excerpt: string; flag_density?: number }>;
  co_firing?: Array<{ concepts: string[]; co_rate: number }>;
};

export type MeldReview = {
  reviewer: string;
  decision: "approve" | "reject" | "request_changes" | "escalate";
  comment?: string;
  decided_at: string;
};

export type Meld = {
  id: string;
  state: MeldState;
  source: MeldSource;
  target_pack: string;
  protection_level: ProtectionLevel;
  candidates: MeldCandidate[];
  structural_ops: MeldStructuralOp[];
  impact?: MeldImpact;
  evidence?: MeldEvidence;
  reviews: MeldReview[];
  created_at: string;
  updated_at: string;
};

// ---------- Runs ----------

export type RunType = "training" | "simplex" | "calibration" | "eval";
export type RunStatus = "pending" | "running" | "succeeded" | "failed" | "killed";

export type Run = {
  id: string;
  type: RunType;
  status: RunStatus;
  substrate: string;
  target_pack?: string;
  config: Record<string, unknown>;
  started_at?: string;
  ended_at?: string;
  current_step?: string;
  progress?: { current: number; total: number; eta_seconds?: number };
  exit_code?: number;
  log_path?: string;
  output_artifacts: string[];
};

// ---------- Docs ----------

export type DocFrontmatter = Record<string, unknown>;

export type DocHeading = {
  level: number;
  text: string;
  anchor: string;
};

export type Doc = {
  path: string; // relative path from indexed root
  title: string;
  folder: string;
  frontmatter: DocFrontmatter;
  body: string;
  headings: DocHeading[];
  inbound_links: string[];
  outbound_links: string[];
  updated_at: string;
};

export type DocTreeNode = {
  name: string;
  path: string;
  is_directory: boolean;
  children?: DocTreeNode[];
};

// ---------- Settings / env ----------

export type EnvironmentReport = {
  hatcatdev_url: string;
  python: {
    active_env: "venv" | "conda" | "none";
    env_name?: string;
    version?: string;
  };
  node: { version: string };
  hf_cache: { path: string; size_bytes: number };
  gpu?: { name: string; total_memory_bytes: number; available_memory_bytes: number };
  default_substrate?: string;
  admin_token_configured: boolean;
};

export type HealthStatus = {
  status: "ok" | "degraded" | "down";
  version: string;
  uptime_seconds: number;
};

// ---------- Registry ----------

export type LensPackRegistryEntry = {
  id: string;
  path: string;
  substrate: string;
  concept_pack: string;
  version: string;
};

export type Registry = {
  lens_packs: LensPackRegistryEntry[];
  concept_packs: Array<{ name: string; path: string; version: string }>;
};
