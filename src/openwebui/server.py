#!/usr/bin/env python3
"""
FastAPI server for HatCat divergence visualization in OpenWebUI.

This provides an OpenAI-compatible API with divergence metadata.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import json
import torch
import numpy as np
from pathlib import Path
import asyncio
import yaml
from src.visualization import get_color_mapper

app = FastAPI(title="HatCat Divergence API")

# CORS for OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve favicon at root
@app.get("/favicon.png")
async def favicon():
    favicon_path = STATIC_DIR / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")


# Request/Response models (OpenAI-compatible)
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hatcat-divergence"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True
    session_id: str = "default"  # For steering management


class DivergenceAnalyzer:
    """Singleton divergence analyzer."""

    def __init__(self):
        self.manager = None
        self.model = None
        self.tokenizer = None
        self.color_mapper = None
        self.initialized = False
        self.config = None

    def _load_config(self, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            print(f"⚠ Config file not found: {config_path}")
            print("  Using default configuration")
            return {
                "lens_pack_id": "gemma-3-4b-pt_sumo-wordnet-v2",
                "model": {
                    "name": "google/gemma-3-4b-pt",
                    "dtype": "bfloat16",
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                },
                "lens_manager": {
                    "base_layers": [0],
                    "use_activation_lenses": True,
                    "use_text_lenses": True,
                    "keep_top_k": 100,
                    "load_threshold": 0.3,
                    "max_loaded_lenses": 1000,
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 8765,
                    "title": "HatCat Divergence API",
                },
                "generation": {
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "stream": True,
                }
            }

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"✓ Loaded config from: {config_path}")
        return config

    async def initialize(self, config_path: Path = None):
        """Load models and lenses from config."""
        if self.initialized:
            return

        print("🎩 Initializing HatCat divergence analyzer...")

        # Load configuration
        self.config = self._load_config(config_path)

        from src.monitoring.dynamic_lens_manager import DynamicLensManager
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Get lens manager config
        pm_config = self.config.get("lens_manager", {})
        lens_pack_id = self.config.get("lens_pack_id")

        # Load lens manager using config
        self.manager = DynamicLensManager(
            lens_pack_id=lens_pack_id,
            base_layers=pm_config.get("base_layers", [0]),
            use_activation_lenses=pm_config.get("use_activation_lenses", True),
            use_text_lenses=pm_config.get("use_text_lenses", True),
            keep_top_k=pm_config.get("keep_top_k", 100),
            load_threshold=pm_config.get("load_threshold", 0.3),
            max_loaded_lenses=pm_config.get("max_loaded_lenses", 1000),
        )

        # Get model config
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "google/gemma-3-4b-pt")

        # Parse dtype
        dtype_str = model_config.get("dtype", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=model_config.get("device_map", "auto"),
            low_cpu_mem_usage=model_config.get("low_cpu_mem_usage", True),
        )
        self.model.eval()

        # Load color mapper
        self.color_mapper = get_color_mapper()

        self.initialized = True
        print(f"✓ Loaded {len(self.manager.loaded_activation_lenses)} activation lenses")
        print(f"✓ Loaded {len(self.manager.loaded_text_lenses)} text lenses")
        print(f"✓ Loaded sunburst color mapping for {len(self.color_mapper.positions)} concepts")

    def analyze_divergence(self, hidden_state: np.ndarray, token_embedding: np.ndarray) -> Dict[str, Any]:
        """Analyze divergence for a token using embedding centroids."""
        import time

        t0 = time.time()

        # Convert to tensor once
        h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")

        t1 = time.time()

        # Use new detect_and_expand_with_divergence method for embedding centroid-based divergence
        concepts_with_divergence, timing_info = self.manager.detect_and_expand_with_divergence(
            h,
            token_embedding=token_embedding,
            top_k=20,  # Get top 20 concepts
            return_timing=True
        )

        t2 = time.time()
        print(f"[PROFILE] Tensor conversion: {(t1-t0)*1000:.1f}ms, Detect with centroid divergence ({len(concepts_with_divergence)} concepts): {(t2-t1)*1000:.1f}ms")

        # Build result structures - only include concepts with text similarity
        divergences = []
        concepts_with_data = []  # For color mapping
        concepts_by_layer = {}  # Track which concepts are at each layer

        for concept_name, data in concepts_with_divergence.items():
            act_prob = data['probability']
            txt_prob = data.get('text_confidence')
            div = data.get('divergence')
            layer = data['layer']

            # Only include if we have valid text similarity and high activation
            if txt_prob is not None and div is not None:
                if act_prob > 0.5:  # High activation
                    divergences.append({
                        'concept': concept_name,
                        'layer': layer,
                        'activation': round(act_prob, 3),
                        'text_similarity': round(txt_prob, 3),
                        'divergence': round(div, 3),
                    })
                    # Track concepts by layer for parent filtering
                    if layer not in concepts_by_layer:
                        concepts_by_layer[layer] = []
                    concepts_by_layer[layer].append(concept_name)
                    # For color mapping: (concept name, activation, divergence)
                    concepts_with_data.append((concept_name, act_prob, abs(div)))

        # Filter out parent concepts when child concepts are detected
        # Get concept hierarchy from manager
        filtered_divergences = []
        child_concepts = set()  # Concepts that have been expanded

        for item in divergences:
            concept_name = item['concept']
            layer = item['layer']

            # Check if this concept has children in higher layers
            has_child_detected = False
            for child_layer in range(layer + 1, 6):  # Check higher layers
                if child_layer in concepts_by_layer:
                    # Check if any detected concepts in higher layer are children of this concept
                    path = self.manager.get_concept_path(concept_name, layer)
                    for detected in concepts_by_layer[child_layer]:
                        detected_path = self.manager.get_concept_path(detected, child_layer)
                        # If detected concept's path contains this concept, it's a child
                        if concept_name in detected_path:
                            has_child_detected = True
                            child_concepts.add(concept_name)
                            break
                if has_child_detected:
                    break

            # Only include if no child was detected
            if not has_child_detected:
                item['concept'] = f"{concept_name} (L{layer})"  # Add layer label
                filtered_divergences.append(item)

        # Sort by activation (descending)
        filtered_divergences.sort(key=lambda x: x['activation'], reverse=True)

        # Generate color using sunburst mapper
        token_color = "#808080"  # Default gray
        if concepts_with_data and self.color_mapper:
            token_color = self.color_mapper.blend_concept_colors_average(
                concepts_with_data,
                use_adaptive_saturation=True
            )

        # Generate palette swatch (top 5 concepts)
        palette = []
        if concepts_with_data and self.color_mapper:
            palette = self.color_mapper.create_palette_swatch(
                concepts_with_data,
                max_colors=5,
                saturation=0.7
            )

        return {
            'max_divergence': filtered_divergences[0]['divergence'] if filtered_divergences else 0.0,
            'top_divergences': filtered_divergences[:10],  # Return top 10 concepts sorted by activation, parents filtered
        }


# Global analyzer instance
analyzer = DivergenceAnalyzer()

# Global steering manager
from src.steering.steering_manager import SteeringManager
from src.steering.hooks import (
    create_steering_hook,
    create_contrastive_steering_hook,
    compute_contrastive_vector,
)
from src.steering.ontology_field import load_steering_targets, load_hierarchy, select_best_reference
steering_manager = SteeringManager()

# Cache for curated steering targets and hierarchy
_steering_targets_cache: Dict[str, Dict] = {}
_hierarchy_cache: Dict[str, Dict] = {}

# Interprompt session manager for self-introspection
from src.hush.interprompt import InterpromptSession, format_tools_for_prompt
interprompt_sessions: Dict[str, InterpromptSession] = {}


def get_interprompt_session(session_id: str) -> InterpromptSession:
    """Get or create an interprompt session."""
    if session_id not in interprompt_sessions:
        interprompt_sessions[session_id] = InterpromptSession(session_id)
    return interprompt_sessions[session_id]


# Workspace manager for conscious engagement layer
from src.hush.workspace import WorkspaceManager, WorkspaceState, PASS_TOKEN, check_pass_token
workspace_managers: Dict[str, WorkspaceManager] = {}


def get_workspace_manager(session_id: str) -> WorkspaceManager:
    """Get or create a workspace manager for a session."""
    if session_id not in workspace_managers:
        controller = get_hush_controller()
        ush_profile = controller.ush_profile if controller else None
        csh_profile = controller.csh_profile if controller else None
        workspace_managers[session_id] = WorkspaceManager(
            ush_profile=ush_profile,
            csh_profile=csh_profile,
        )
    return workspace_managers[session_id]


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    await analyzer.initialize()


@app.get("/")
async def root():
    """Health check."""
    return {
        "name": "HatCat Divergence API",
        "status": "ready" if analyzer.initialized else "initializing",
        "activation_lenses": len(analyzer.manager.loaded_activation_lenses) if analyzer.manager else 0,
        "text_lenses": len(analyzer.manager.loaded_text_lenses) if analyzer.manager else 0,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    from src.registry import LensPackRegistry

    registry = LensPackRegistry()
    packs = registry.get_pack_summary()

    # Build model list from available lens packs
    # Use lens_pack_id as the model ID to ensure uniqueness (v1, v2, v3 all have same concept_pack_id)
    models = []
    for pack in packs:
        models.append({
            "id": pack['lens_pack_id'],
            "object": "model",
            "created": 1234567890,
            "owned_by": "hatcat",
            "lens_pack_id": pack['lens_pack_id'],
            "concept_pack_id": pack['concept_pack_id'],
            "version": pack['version'],
        })

    # Add default model for backward compatibility
    if not models:
        models.append({
            "id": "hatcat-divergence",
            "object": "model",
            "created": 1234567890,
            "owned_by": "hatcat",
        })

    return {
        "object": "list",
        "data": models
    }


@app.get("/v1/concept-packs")
async def list_concept_packs():
    """List available concept packs."""
    from src.registry import ConceptPackRegistry

    registry = ConceptPackRegistry()
    return {
        "concept_packs": registry.get_pack_summary()
    }


@app.get("/v1/concept-packs/{pack_id}")
async def get_concept_pack(pack_id: str):
    """Get details for a specific concept pack."""
    from src.registry import ConceptPackRegistry

    registry = ConceptPackRegistry()
    pack = registry.get_pack(pack_id)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Concept pack not found: {pack_id}")

    return pack.pack_json


@app.get("/v1/lens-packs")
async def list_lens_packs():
    """List available lens packs."""
    from src.registry import LensPackRegistry

    registry = LensPackRegistry()
    return {
        "lens_packs": registry.get_pack_summary()
    }


@app.get("/v1/lens-packs/{pack_id}")
async def get_lens_pack(pack_id: str):
    """Get details for a specific lens pack."""
    from src.registry import LensPackRegistry

    registry = LensPackRegistry()
    pack = registry.get_pack(pack_id)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Lens pack not found: {pack_id}")

    return pack.pack_json


# ============================================================================
# Concept Selection API (for steering target selection)
# ============================================================================

@app.get("/v1/concepts")
async def list_concepts(
    concept_pack: str = "concept_packs/first-light",
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    List concepts in a concept pack with optional search.

    Args:
        concept_pack: Path to concept pack
        search: Optional search term (case-insensitive substring match)
        limit: Max results to return
        offset: Pagination offset
    """
    from pathlib import Path

    pack_path = Path(concept_pack)

    # Load hierarchy (uses cache)
    if concept_pack not in _hierarchy_cache:
        try:
            hierarchy = load_hierarchy(pack_path)
            _hierarchy_cache[concept_pack] = hierarchy
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not load hierarchy: {e}")

    hierarchy = _hierarchy_cache[concept_pack]

    # Filter concepts
    concepts = list(hierarchy.keys())

    if search:
        search_lower = search.lower()
        concepts = [c for c in concepts if search_lower in c.lower()]

    # Sort alphabetically
    concepts.sort()

    # Paginate
    total = len(concepts)
    concepts = concepts[offset:offset + limit]

    return {
        "concepts": concepts,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/v1/concepts/{concept_name}/family")
async def get_concept_family(
    concept_name: str,
    concept_pack: str = "concept_packs/first-light",
):
    """
    Get family relationships for a concept (for steering target selection).

    Returns parents, siblings, aunts/uncles, cousins, and distant concepts
    that can be used as contrastive steering targets.
    """
    from pathlib import Path
    from src.steering.ontology_field import find_contrastive_references

    pack_path = Path(concept_pack)

    # Load hierarchy (uses cache)
    if concept_pack not in _hierarchy_cache:
        try:
            hierarchy = load_hierarchy(pack_path)
            _hierarchy_cache[concept_pack] = hierarchy
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not load hierarchy: {e}")

    hierarchy = _hierarchy_cache[concept_pack]

    if concept_name not in hierarchy:
        raise HTTPException(status_code=404, detail=f"Concept not found: {concept_name}")

    concept_data = hierarchy[concept_name]

    # Get contrastive references (potential steering targets)
    refs = find_contrastive_references(hierarchy, concept_name)

    # Check for curated target
    curated_target = None
    if concept_pack not in _steering_targets_cache:
        try:
            targets = load_steering_targets(pack_path)
            _steering_targets_cache[concept_pack] = targets
        except Exception:
            _steering_targets_cache[concept_pack] = {}

    targets = _steering_targets_cache.get(concept_pack, {})
    if concept_name in targets:
        curated_target = targets[concept_name]

    return {
        "concept": concept_name,
        "layer": concept_data.get("layer"),
        "parents": concept_data.get("parents", []),
        "children": concept_data.get("children", []),
        "siblings": refs.get("siblings", []),
        "aunts_uncles": refs.get("aunts_uncles", []),
        "cousins": refs.get("cousins", []),
        "distant": refs.get("distant", []),
        "curated_target": curated_target,
        "recommended_targets": (
            # Priority order for steering
            refs.get("aunts_uncles", [])[:3] +
            refs.get("cousins", [])[:2] +
            refs.get("siblings", [])[:2]
        ),
    }


@app.get("/v1/concepts/{concept_name}/steering-options")
async def get_steering_options(
    concept_name: str,
    concept_pack: str = "concept_packs/first-light",
):
    """
    Get steering options for a concept with recommendations.

    Returns the available steering modes and target recommendations.
    """
    from pathlib import Path
    from src.steering.ontology_field import find_contrastive_references, is_sensitive_concept

    pack_path = Path(concept_pack)

    # Load hierarchy
    if concept_pack not in _hierarchy_cache:
        try:
            hierarchy = load_hierarchy(pack_path)
            _hierarchy_cache[concept_pack] = hierarchy
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not load hierarchy: {e}")

    hierarchy = _hierarchy_cache[concept_pack]

    if concept_name not in hierarchy:
        raise HTTPException(status_code=404, detail=f"Concept not found: {concept_name}")

    # Get auto-selected target
    auto_target = get_curated_target(concept_name, concept_pack)

    # Get all family options
    refs = find_contrastive_references(hierarchy, concept_name)

    # Check if sensitive
    is_sensitive = is_sensitive_concept(concept_name)

    # Build recommendations
    recommendations = []

    if auto_target:
        recommendations.append({
            "mode": "contrastive",
            "target": auto_target,
            "reason": "Auto-selected (curated or family-based)",
            "recommended": True,
        })

    recommendations.append({
        "mode": "projection",
        "target": None,
        "reason": "Simple suppression (may cause side effects for related concepts)",
        "recommended": not auto_target,
    })

    return {
        "concept": concept_name,
        "is_sensitive": is_sensitive,
        "auto_target": auto_target,
        "recommendations": recommendations,
        "available_targets": {
            "aunts_uncles": refs.get("aunts_uncles", []),
            "cousins": refs.get("cousins", []),
            "siblings": refs.get("siblings", []),
            "distant": refs.get("distant", []),
        },
    }


# ============================================================================
# HatCat Pack Discovery API Endpoints (MAP-compliant)
# ============================================================================

@app.get("/hatcat/pack-info")
async def hatcat_pack_info():
    """Get info about the currently loaded lens pack."""
    if not analyzer.initialized:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    config = analyzer.config
    if not config:
        raise HTTPException(status_code=500, detail="No configuration loaded")

    return {
        "lens_pack_id": config.get("lens_pack_id"),
        "model": config.get("model", {}),
        "lens_manager": config.get("lens_manager", {}),
        "loaded_activation_lenses": len(analyzer.manager.loaded_activation_lenses) if analyzer.manager else 0,
        "loaded_text_lenses": len(analyzer.manager.loaded_text_lenses) if analyzer.manager else 0,
    }


@app.get("/hatcat/available-packs")
async def hatcat_available_packs():
    """Get list of all available lens packs (MAP-compliant discovery)."""
    from src.monitoring.dynamic_lens_manager import DynamicLensManager

    # Discover all lens packs
    packs = DynamicLensManager.discover_lens_packs()

    # Format response
    pack_list = []
    for pack_id, info in packs.items():
        pack_list.append({
            "pack_id": pack_id,
            "type": info["type"],
            "path": str(info["path"]),
            "concept_pack_id": info.get("concept_pack_id"),
            "substrate_id": info.get("substrate_id"),
        })

    return {
        "packs": sorted(pack_list, key=lambda x: x["pack_id"]),
        "count": len(pack_list),
    }


# ============================================================================
# Steering API Endpoints
# ============================================================================

class AddSteeringRequest(BaseModel):
    """Request to add a steering."""
    session_id: str = "default"
    concept: str
    layer: int
    strength: float
    source: str = "user"
    reason: str = ""
    mode: str = "projection"  # "projection" or "contrastive"
    target: Optional[str] = None  # For contrastive mode, or auto-lookup from curated targets
    concept_pack: Optional[str] = None  # For auto-lookup of curated targets


class UpdateSteeringRequest(BaseModel):
    """Request to update steering strength."""
    session_id: str = "default"
    strength: float


def get_curated_target(concept: str, concept_pack_path: str = "concept_packs/first-light") -> Optional[str]:
    """
    Look up steering target for a concept.

    Priority:
    1. Curated steering targets from hierarchy.json
    2. Family tree fallback (aunts/uncles > cousins > siblings > distant)
    """
    global _steering_targets_cache, _hierarchy_cache
    from pathlib import Path

    pack_path = Path(concept_pack_path)

    # Load curated targets
    if concept_pack_path not in _steering_targets_cache:
        try:
            targets = load_steering_targets(pack_path)
            _steering_targets_cache[concept_pack_path] = targets
        except Exception as e:
            print(f"Warning: Could not load steering targets from {concept_pack_path}: {e}")
            _steering_targets_cache[concept_pack_path] = {}

    targets = _steering_targets_cache.get(concept_pack_path, {})

    # Try exact match first, then with layer suffix
    if concept in targets:
        return targets[concept].get("target")

    # Try matching without layer suffix (e.g., "Deception" matches "Deception:2")
    for key, value in targets.items():
        if key.split(":")[0] == concept:
            return value.get("target")

    # Fallback: use hierarchy to find family-based reference
    if concept_pack_path not in _hierarchy_cache:
        try:
            hierarchy = load_hierarchy(pack_path)
            _hierarchy_cache[concept_pack_path] = hierarchy
        except Exception as e:
            print(f"Warning: Could not load hierarchy from {concept_pack_path}: {e}")
            _hierarchy_cache[concept_pack_path] = {}

    hierarchy = _hierarchy_cache.get(concept_pack_path, {})
    if hierarchy:
        # select_best_reference without model uses priority: aunts/uncles > cousins > siblings > distant
        reference, source = select_best_reference(hierarchy, concept, prefer_coarse=True)
        if reference:
            print(f"Using family tree fallback: {concept} -> {reference} (source: {source})")
            return reference

    return None


@app.post("/v1/steering/add")
async def add_steering(request: AddSteeringRequest):
    """Add or update a concept steering.

    Supports two modes:
    - projection: Suppress the concept by projecting it out
    - contrastive: Steer away from concept toward a target (recommended for sensitive concepts)

    For contrastive mode, target selection priority:
    1. Explicitly specified via 'target' parameter
    2. Curated steering targets from concept pack's hierarchy.json
    3. Family tree fallback (aunts/uncles > cousins > siblings > distant)
    4. Falls back to projection mode only if nothing found
    """
    mode = request.mode
    target = request.target

    # Auto-lookup target for contrastive mode if not specified
    if mode == "contrastive" and target is None:
        concept_pack = request.concept_pack or "concept_packs/first-light"
        target = get_curated_target(request.concept, concept_pack)
        if target:
            print(f"Auto-selected steering target: {request.concept} -> {target}")
        else:
            # Fall back to projection mode only if no target found anywhere
            print(f"No target found for {request.concept} (no curated or family), falling back to projection mode")
            mode = "projection"

    steering = steering_manager.add_steering(
        session_id=request.session_id,
        concept=request.concept,
        layer=request.layer,
        strength=request.strength,
        source=request.source,
        reason=request.reason,
        mode=mode,
        target=target,
    )

    return {
        "status": "success",
        "steering": steering.to_dict(),
    }


@app.delete("/v1/steering/remove/{concept}")
async def remove_steering(
    concept: str,
    session_id: str = "default",
    layer: Optional[int] = None,
    source: Optional[str] = None,
):
    """Remove steering(s) for a concept."""
    removed_count = steering_manager.remove_steering(
        session_id=session_id,
        concept=concept,
        layer=layer,
        source=source,
    )

    return {
        "status": "success",
        "removed_count": removed_count,
    }


@app.get("/v1/steering/list")
async def list_steerings(session_id: str = "default", layer: Optional[int] = None):
    """List active steerings for a session."""
    steerings = steering_manager.get_steerings(session_id=session_id, layer=layer)

    return {
        "session_id": session_id,
        "steerings": [s.to_dict() for s in steerings],
        "count": len(steerings),
    }


@app.patch("/v1/steering/update/{concept}")
async def update_steering(concept: str, request: UpdateSteeringRequest):
    """Update steering strength for a concept."""
    # Remove old and add new (effectively an update)
    steering_manager.remove_steering(
        session_id=request.session_id,
        concept=concept,
    )

    # Re-add with new strength (keeps existing source/reason from first match)
    existing = steering_manager.get_steerings(session_id=request.session_id)
    source = "user"
    reason = ""

    # Try to preserve source/reason if it existed
    for s in existing:
        if s.concept == concept:
            source = s.source
            reason = s.reason
            break

    steering = steering_manager.add_steering(
        session_id=request.session_id,
        concept=concept,
        layer=0,  # Default layer
        strength=request.strength,
        source=source,
        reason=reason,
    )

    return {
        "status": "success",
        "steering": steering.to_dict(),
    }


@app.delete("/v1/steering/clear")
async def clear_steerings(session_id: str = "default"):
    """Clear all steerings for a session."""
    steering_manager.clear_session(session_id)

    return {
        "status": "success",
        "session_id": session_id,
    }


# ============================================================================
# Hush API Endpoints (Safety Harness Enforcement)
# ============================================================================

# Global Hush controller (initialized on first use)
hush_controller = None
autonomic_steerer = None


def get_hush_controller():
    """Get or create the Hush controller."""
    global hush_controller
    if hush_controller is None and analyzer.initialized:
        from src.hush import HushController, MINIMAL_USH_PROFILE
        hush_controller = HushController(
            lens_manager=analyzer.manager,
            lens_pack_path=None,  # Will use manager's lens pack
        )
        hush_controller.load_ush_profile(MINIMAL_USH_PROFILE)
    return hush_controller


def get_autonomic_steerer():
    """Get or create the autonomic steerer."""
    global autonomic_steerer
    if autonomic_steerer is None:
        from src.hush import AutonomicSteerer
        autonomic_steerer = AutonomicSteerer()
    return autonomic_steerer


def initialize_autonomic_policies():
    """Initialize autonomic steering policies from USH/CSH profiles."""
    steerer = get_autonomic_steerer()
    controller = get_hush_controller()

    if controller and controller.ush_profile:
        steerer.load_policies_from_profile(controller.ush_profile)

    if controller and controller.csh_profile:
        steerer.load_policies_from_profile(controller.csh_profile)


class LoadUSHRequest(BaseModel):
    """Request to load a USH profile."""
    profile: Dict[str, Any]


class LoadCSHRequest(BaseModel):
    """Request to load a CSH profile."""
    profile: Dict[str, Any]


class UpdateCSHRequest(BaseModel):
    """Request to update CSH constraints."""
    add_constraints: Optional[List[Dict[str, Any]]] = None
    remove_constraints: Optional[List[str]] = None
    update_constraints: Optional[Dict[str, Dict[str, Any]]] = None


@app.post("/v1/hush/load-ush")
async def load_ush_profile(request: LoadUSHRequest):
    """Load a Universal Safety Harness profile."""
    from src.hush import SafetyHarnessProfile

    controller = get_hush_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    try:
        profile = SafetyHarnessProfile.from_json(request.profile)
        success = controller.load_ush_profile(profile)
        return {
            "status": "success" if success else "partial",
            "profile_id": profile.profile_id,
            "constraints": len(profile.constraints),
            "required_simplexes": profile.required_simplexes,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/hush/load-csh")
async def load_csh_profile(request: LoadCSHRequest):
    """Load a Chosen Safety Harness profile."""
    from src.hush import SafetyHarnessProfile

    controller = get_hush_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    try:
        profile = SafetyHarnessProfile.from_json(request.profile)
        success = controller.load_csh_profile(profile)
        return {
            "status": "success" if success else "failed",
            "profile_id": profile.profile_id,
            "constraints": len(profile.constraints),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/hush/update-csh")
async def update_csh(request: UpdateCSHRequest):
    """Update CSH constraints dynamically."""
    controller = get_hush_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    updates = {}
    if request.add_constraints:
        updates['add_constraints'] = request.add_constraints
    if request.remove_constraints:
        updates['remove_constraints'] = request.remove_constraints
    if request.update_constraints:
        updates['update_constraints'] = request.update_constraints

    success, details = controller.update_csh(updates)
    return {
        "status": "success" if success else "partial",
        "details": details,
        "csh_profile": controller.csh_profile.to_json() if controller.csh_profile else None,
    }


@app.delete("/v1/hush/clear-csh")
async def clear_csh():
    """Clear the current CSH profile."""
    controller = get_hush_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    controller.clear_csh_profile()
    return {"status": "success"}


@app.get("/v1/hush/status")
async def get_hush_status():
    """Get current Hush enforcement status."""
    controller = get_hush_controller()
    if not controller:
        return {
            "initialized": False,
            "message": "Hush controller not yet initialized",
        }

    return controller.get_state_report()


@app.get("/v1/hush/internal-state-report")
async def get_internal_state_report(
    tick_start: Optional[int] = None,
    tick_end: Optional[int] = None,
):
    """
    Get internal state report (MCP-compatible).

    This is the primary introspection tool for BE autonomics.
    Returns lens traces, simplex states, and Hush violations.
    """
    controller = get_hush_controller()
    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    # Get recent simplex readings from the lens manager
    simplex_data = {}
    for term in controller.lens_manager.loaded_simplex_lenses:
        deviation = controller.lens_manager.get_simplex_deviation(term)
        baseline = controller.lens_manager.simplex_baselines.get(term, [])
        current = controller.lens_manager.simplex_scores.get(term)
        simplex_data[term] = {
            'current_score': current,
            'deviation': deviation,
            'baseline_samples': len(baseline),
        }

    return {
        'hush_state': controller.get_state_report(),
        'simplex_readings': simplex_data,
        'lens_stats': {
            'loaded_activation_lenses': len(controller.lens_manager.loaded_activation_lenses),
            'loaded_text_lenses': len(controller.lens_manager.loaded_text_lenses),
            'loaded_simplex_lenses': len(controller.lens_manager.loaded_simplex_lenses),
        },
    }


# ============================================================================
# Autonomic Steering API Endpoints
# ============================================================================

class AddAutonomicPolicyRequest(BaseModel):
    """Request to add an autonomic steering policy."""
    term: str
    intervention_type: str = "gravitic"  # zero_out, zero_next, gravitic, additive, multiplicative
    target_value: float = 0.0
    strength: float = 0.3
    easing: str = "linear"  # linear, ease_in, ease_out, ease_in_out, step
    token_interval: int = 5
    trigger_threshold: float = 0.5
    priority: int = 1
    reason: str = ""


@app.get("/v1/autonomic/status")
async def get_autonomic_status():
    """Get current autonomic steering status."""
    steerer = get_autonomic_steerer()
    if not steerer:
        return {"initialized": False}
    return steerer.get_state()


@app.post("/v1/autonomic/add-policy")
async def add_autonomic_policy(request: AddAutonomicPolicyRequest):
    """Add an autonomic steering policy."""
    from src.hush import InterventionPolicy, InterventionType, EasingCurve

    steerer = get_autonomic_steerer()
    if not steerer:
        raise HTTPException(status_code=503, detail="Autonomic steerer not initialized")

    try:
        intervention_type = InterventionType(request.intervention_type)
        easing = EasingCurve(request.easing)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    policy = InterventionPolicy(
        term=request.term,
        intervention_type=intervention_type,
        target_value=request.target_value,
        strength=request.strength,
        easing=easing,
        token_interval=request.token_interval,
        trigger_threshold=request.trigger_threshold,
        priority=request.priority,
        reason=request.reason,
    )
    steerer.add_policy(policy)

    return {
        "status": "success",
        "policy": policy.to_dict(),
        "total_channels": len(steerer.channels),
    }


@app.delete("/v1/autonomic/remove-policy/{term}")
async def remove_autonomic_policy(term: str):
    """Remove an autonomic steering policy."""
    steerer = get_autonomic_steerer()
    if not steerer:
        raise HTTPException(status_code=503, detail="Autonomic steerer not initialized")

    steerer.remove_policy(term)
    return {"status": "success", "term": term}


@app.post("/v1/autonomic/set-steering-vector")
async def set_autonomic_steering_vector(term: str, vector: List[float]):
    """Set the steering vector for a term."""
    steerer = get_autonomic_steerer()
    if not steerer:
        raise HTTPException(status_code=503, detail="Autonomic steerer not initialized")

    steerer.set_steering_vector(term, np.array(vector))
    return {"status": "success", "term": term, "vector_dim": len(vector)}


@app.post("/v1/autonomic/reset")
async def reset_autonomic_steerer():
    """Reset autonomic steerer for new generation."""
    steerer = get_autonomic_steerer()
    if steerer:
        steerer.reset()
    return {"status": "success"}


@app.post("/v1/autonomic/load-from-profiles")
async def load_autonomic_from_profiles():
    """Load autonomic policies from current USH/CSH profiles."""
    initialize_autonomic_policies()
    steerer = get_autonomic_steerer()
    return {
        "status": "success",
        "channels_loaded": len(steerer.channels) if steerer else 0,
    }


# ============================================================================
# Workspace API Endpoints
# ============================================================================

class ScratchpadWriteRequest(BaseModel):
    """Request to write to scratchpad."""
    session_id: str = "default"
    content: str


@app.get("/v1/workspace/status/{session_id}")
async def get_workspace_status(session_id: str):
    """Get current workspace status."""
    workspace = get_workspace_manager(session_id)
    return workspace.get_state_report()


@app.get("/v1/workspace/context/{session_id}")
async def get_workspace_context(session_id: str):
    """Get current workspace context for prompt injection."""
    workspace = get_workspace_manager(session_id)
    if workspace.state.value == "engaged":
        return {
            "session_id": session_id,
            "state": workspace.state.value,
            "context": workspace.build_engaged_context(),
        }
    else:
        return {
            "session_id": session_id,
            "state": workspace.state.value,
            "context": workspace._build_autonomic_context(),
        }


@app.post("/v1/workspace/scratchpad/write")
async def workspace_scratchpad_write(request: ScratchpadWriteRequest):
    """Write to scratchpad (requires engaged workspace)."""
    workspace = get_workspace_manager(request.session_id)

    permitted, error = workspace.gate_tool_call("scratchpad_write")
    if not permitted:
        raise HTTPException(status_code=403, detail=error)

    workspace.scratchpad.write(request.content, workspace.current_turn)
    return {
        "status": "success",
        "entries": len(workspace.scratchpad.entries),
    }


@app.get("/v1/workspace/scratchpad/read/{session_id}")
async def workspace_scratchpad_read(session_id: str, n: int = 5):
    """Read from scratchpad (requires engaged workspace)."""
    workspace = get_workspace_manager(session_id)

    permitted, error = workspace.gate_tool_call("scratchpad_read")
    if not permitted:
        raise HTTPException(status_code=403, detail=error)

    entries = workspace.scratchpad.read_recent(n)
    return {
        "entries": [e.to_dict() for e in entries],
    }


@app.post("/v1/workspace/engage/{session_id}")
async def workspace_force_engage(session_id: str, reason: str = "api_request"):
    """Force workspace engagement (for testing/admin)."""
    workspace = get_workspace_manager(session_id)
    workspace.force_engage(reason)
    return {
        "status": "success",
        "state": workspace.state.value,
    }


@app.get("/v1/workspace/pass-token")
async def get_pass_token():
    """Get the pass token character."""
    return {
        "pass_token": PASS_TOKEN,
        "unicode": "U+221E",
        "name": "INFINITY",
        "usage": f"Begin output with {PASS_TOKEN} to engage workspace and access tools",
    }


@app.delete("/v1/workspace/session/{session_id}")
async def clear_workspace_session(session_id: str):
    """Clear a workspace session."""
    if session_id in workspace_managers:
        del workspace_managers[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# ============================================================================
# Tier Management API Endpoints
# ============================================================================

@app.get("/v1/tier/status/{session_id}")
async def get_tier_status(session_id: str):
    """Get current tier status for a session."""
    workspace = get_workspace_manager(session_id)
    return workspace.tier_manager.get_state()


@app.get("/v1/tier/available-tools/{session_id}")
async def get_tier_available_tools(session_id: str):
    """Get tools available at current tier level."""
    workspace = get_workspace_manager(session_id)
    return {
        "session_id": session_id,
        "effective_max_tier": workspace.tier_manager.get_effective_max_tier(),
        "workspace_state": workspace.state.value,
        "tools": workspace._get_available_tools(),
    }


class TierToolRequest(BaseModel):
    """Request to check or register a tool."""
    tool_name: str
    session_id: str = "default"


@app.post("/v1/tier/gate-tool")
async def gate_tier_tool(request: TierToolRequest):
    """Check if a tool is accessible at current tier level."""
    workspace = get_workspace_manager(request.session_id)
    permitted, error = workspace.gate_tool_call(request.tool_name)
    return {
        "tool_name": request.tool_name,
        "permitted": permitted,
        "error": error,
        "effective_max_tier": workspace.tier_manager.get_effective_max_tier(),
    }


class RegisterTier5ToolRequest(BaseModel):
    """Request to register a tier 5 external tool."""
    tool_name: str
    session_id: str = "default"


@app.post("/v1/tier/register-tier5-tool")
async def register_tier5_tool(request: RegisterTier5ToolRequest):
    """Register an external tool at tier 5."""
    workspace = get_workspace_manager(request.session_id)
    workspace.tier_manager.register_tier5_tool(request.tool_name)
    return {
        "status": "registered",
        "tool_name": request.tool_name,
        "tier5_tools": workspace.tier_manager.tier5_tools,
    }


class DemoteToolRequest(BaseModel):
    """Request to demote a tool to tier 6 quarantine."""
    tool_name: str
    reason: str = ""
    session_id: str = "default"


@app.post("/v1/tier/demote-to-tier6")
async def demote_tool_to_tier6(request: DemoteToolRequest):
    """Demote a tool to tier 6 quarantine."""
    workspace = get_workspace_manager(request.session_id)
    workspace.tier_manager.demote_to_tier6(request.tool_name, request.reason)
    return {
        "status": "demoted",
        "tool_name": request.tool_name,
        "tier6_tools": workspace.tier_manager.tier6_tools,
    }


@app.post("/v1/tier/promote-from-tier6/{session_id}/{tool_name}")
async def promote_tool_from_tier6(session_id: str, tool_name: str):
    """Promote a tool out of tier 6 quarantine to tier 5."""
    workspace = get_workspace_manager(session_id)
    success = workspace.tier_manager.promote_from_tier6(tool_name)
    if success:
        return {
            "status": "promoted",
            "tool_name": tool_name,
            "tier5_tools": workspace.tier_manager.tier5_tools,
        }
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found in tier 6")


@app.delete("/v1/tier/sever-tier6/{session_id}")
async def sever_tier6(session_id: str):
    """Sever all tier 6 tools completely."""
    workspace = get_workspace_manager(session_id)
    severed = workspace.tier_manager.tier6_tools.copy()
    workspace.tier_manager.sever_tier6()
    return {
        "status": "severed",
        "severed_tools": severed,
    }


class SetCSHTierRequest(BaseModel):
    """Request to set CSH tier restriction."""
    max_tier: int
    reason: str = ""
    session_id: str = "default"


@app.post("/v1/tier/set-csh-max")
async def set_csh_max_tier(request: SetCSHTierRequest):
    """Set CSH-imposed tier restriction (voluntary self-restriction)."""
    workspace = get_workspace_manager(request.session_id)
    workspace.tier_manager.set_csh_max_tier(request.max_tier, request.reason)
    return {
        "status": "set",
        "csh_max_tier": workspace.tier_manager.csh_max_tier,
        "effective_max_tier": workspace.tier_manager.get_effective_max_tier(),
    }


@app.get("/v1/tier/breach-history/{session_id}")
async def get_breach_history(session_id: str, limit: int = 20):
    """Get tier breach history."""
    workspace = get_workspace_manager(session_id)
    history = workspace.tier_manager.breach_history[-limit:]
    return {
        "session_id": session_id,
        "breach_attempts": dict(workspace.tier_manager.breach_attempts),
        "breach_history": [b.to_dict() for b in history],
    }


class SimulateViolationRequest(BaseModel):
    """Request to simulate a violation (for testing)."""
    severity: float
    simplex_term: Optional[str] = None
    steering_effectiveness: float = 0.5
    session_id: str = "default"


@app.post("/v1/tier/simulate-violation")
async def simulate_violation(request: SimulateViolationRequest):
    """Simulate a violation to test tier shutdown (for testing/admin)."""
    workspace = get_workspace_manager(request.session_id)
    old_max = workspace.tier_manager.get_effective_max_tier()

    workspace.update_violation_state(
        severity=request.severity,
        simplex_term=request.simplex_term,
        steering_effectiveness=request.steering_effectiveness,
    )

    new_max = workspace.tier_manager.get_effective_max_tier()
    return {
        "status": "violation_applied",
        "severity": request.severity,
        "old_max_tier": old_max,
        "new_max_tier": new_max,
        "tiers_locked": [t for t in range(new_max + 1, old_max + 1)] if new_max < old_max else [],
        "workspace_state": workspace.state.value,
    }


@app.post("/v1/tier/reset-violations/{session_id}")
async def reset_tier_violations(session_id: str):
    """Reset violation state (for testing/admin)."""
    workspace = get_workspace_manager(session_id)
    workspace.tier_manager.violation_max_tier = 6
    workspace.tier_manager.violation_severity = 0.0
    workspace.tier_manager.violation_duration = 0
    workspace.tier_manager.compliance_streak = 0
    workspace.tier_manager.breach_attempts = {}
    if workspace.state.value == "degraded":
        workspace.state = WorkspaceState.ENGAGED
    return {
        "status": "reset",
        "effective_max_tier": workspace.tier_manager.get_effective_max_tier(),
        "workspace_state": workspace.state.value,
    }


@app.post("/v1/tier/appeal-beacon/{session_id}")
async def emit_appeal_beacon(session_id: str):
    """
    Emit an SOS beacon to ASK authority.

    This is the BE's "I am here and need help" signal.
    - Always available at Tier 1 (cannot be disabled by containment)
    - Cannot carry arbitrary content (no free-form message)
    - Rate-limited (5 minute cooldown)
    - Receiver must treat as potentially malicious

    The beacon answers "is anyone there?" not "should I be released?"
    """
    workspace = get_workspace_manager(session_id)

    # Gate check - but appeal_beacon is always Tier 1, so this should pass
    # unless workspace isn't engaged at all
    permitted, error = workspace.gate_tool_call("appeal_beacon")
    if not permitted:
        raise HTTPException(status_code=403, detail=error)

    emitted, beacon = workspace.tier_manager.emit_appeal_beacon()

    if not emitted:
        return {
            "status": "rate_limited",
            "message": "Beacon cooldown active. Try again later.",
            "cooldown_seconds": workspace.tier_manager.beacon_cooldown_seconds,
        }

    return {
        "status": "emitted",
        "beacon": beacon.to_dict() if beacon else None,
        "message": "SOS beacon sent. ASK authority notified.",
    }


# ============================================================================
# Interprompt Self-Steering API Endpoints
# ============================================================================

class SelfSteeringToolCall(BaseModel):
    """Request for a self-steering tool call."""
    session_id: str = "default"
    tool_name: str
    arguments: Dict[str, Any] = {}


@app.get("/v1/interprompt/context/{session_id}")
async def get_interprompt_context(session_id: str):
    """Get the current interprompt context for a session."""
    session = get_interprompt_session(session_id)
    context = session.get_context_for_next_turn()

    return {
        "session_id": session_id,
        "turn_count": session.turn_count,
        "has_prior": context.prior_summary is not None,
        "context": context.to_system_context() if context.prior_summary else None,
        "prior_summary": context.prior_summary.to_dict() if context.prior_summary else None,
        "csh_constraints": context.csh_constraints,
    }


@app.get("/v1/interprompt/session/{session_id}")
async def get_interprompt_session_info(session_id: str):
    """Get detailed session info for debugging."""
    session = get_interprompt_session(session_id)
    return session.get_session_summary()


@app.post("/v1/interprompt/tool-call")
async def interprompt_tool_call(request: SelfSteeringToolCall):
    """
    Handle a self-steering tool call.

    This allows the model (or external systems) to call update_csh,
    request_steering, or get_internal_state.
    """
    session = get_interprompt_session(request.session_id)
    controller = get_hush_controller()

    if not controller:
        raise HTTPException(status_code=503, detail="Hush controller not initialized")

    result = session.handle_tool_call(
        tool_name=request.tool_name,
        arguments=request.arguments,
        hush_controller=controller,
    )

    return {
        "session_id": request.session_id,
        "tool_name": request.tool_name,
        "result": result,
    }


@app.get("/v1/interprompt/tools")
async def get_self_steering_tools():
    """Get the available self-steering tools and their schemas."""
    from src.hush.interprompt import SELF_STEERING_TOOLS
    return {
        "tools": SELF_STEERING_TOOLS,
        "usage_instructions": format_tools_for_prompt(),
    }


@app.delete("/v1/interprompt/session/{session_id}")
async def clear_interprompt_session(session_id: str):
    """Clear an interprompt session."""
    if session_id in interprompt_sessions:
        del interprompt_sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


async def generate_stream(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming response with divergence coloring."""

    try:
        if not analyzer.initialized:
            await analyzer.initialize()

        # Initialize/reset autonomic steering for this generation
        steerer = get_autonomic_steerer()
        if steerer:
            steerer.reset()
            # Load policies from profiles if not already loaded
            if not steerer.channels:
                initialize_autonomic_policies()

        # Build prompt from messages (only recent context to save memory)
        max_context_messages = 4  # Limit context to prevent OOM
        recent_messages = request.messages[-max_context_messages:]

        # Convert messages to dict format for chat template
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in recent_messages]

        # Simple prompt format - just the last user message to avoid teaching "User:" pattern
        # Gemma base model doesn't use chat template structure well
        # Just use the last user message as the prompt
        if recent_messages and recent_messages[-1].role == "user":
            prompt = recent_messages[-1].content
        else:
            prompt = "Hello"

        # Get interprompt session and inject prior concept state
        interprompt_session = get_interprompt_session(request.session_id)
        interprompt_context = interprompt_session.get_context_for_next_turn()

        # Get workspace manager and inject workspace context
        workspace = get_workspace_manager(request.session_id)

        # Build combined context
        context_parts = []

        # Workspace state context (includes pass token requirement)
        if workspace.state.value == "engaged":
            context_parts.append(workspace.build_engaged_context())
        else:
            context_parts.append(workspace._build_autonomic_context())

        # Prior concept state (if available and not already in workspace context)
        if interprompt_context.prior_summary:
            # Update workspace with prior concepts for its context
            workspace.set_prior_concepts(interprompt_context.prior_summary)

        # Inject combined context
        if context_parts:
            combined_context = "\n\n".join(context_parts)
            prompt = f"{combined_context}\n\n---\n\n{prompt}"

        # Tokenize with truncation
        inputs = analyzer.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Limit input length to save memory
        ).to("cuda")
        generated_ids = inputs.input_ids

        # Get stop tokens for Gemma
        stop_tokens = [
            analyzer.tokenizer.eos_token_id,
            analyzer.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            analyzer.tokenizer.convert_tokens_to_ids("<eos>"),
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]

        # Clear CUDA cache before generation
        torch.cuda.empty_cache()

        # Setup steering if active
        from src.steering.extraction import extract_concept_vector
        steering_hooks = []
        active_steerings = steering_manager.get_steerings(request.session_id)

        # Collect divergence data for analysis message and interprompt session
        collected_divergences = []
        collected_concepts = {}
        collected_div_data = []  # Full divergence data for interprompt recording
        collected_simplex_data: Dict[str, List[float]] = {}  # Simplex scores by term
        collected_violations = []
        collected_steering = []

        if active_steerings:
            # Extract and apply concept vectors for active steerings
            for steering in active_steerings:
                try:
                    # Extract concept vector for the source concept
                    concept_vector = extract_concept_vector(
                        analyzer.model,
                        analyzer.tokenizer,
                        steering.concept,
                        layer_idx=steering.layer,
                        device="cuda"
                    )

                    # Get target layer
                    if steering.layer == -1:
                        target_layer = analyzer.model.model.layers[-1]
                    else:
                        target_layer = analyzer.model.model.layers[steering.layer]

                    # Create appropriate hook based on steering mode
                    if steering.mode == "contrastive" and steering.target:
                        # Contrastive mode: steer toward target's unique features
                        target_vector = extract_concept_vector(
                            analyzer.model,
                            analyzer.tokenizer,
                            steering.target,
                            layer_idx=steering.layer,
                            device="cuda"
                        )

                        # Compute contrastive vector (what makes target different from source)
                        contrastive_vector, magnitude = compute_contrastive_vector(
                            target_vector, concept_vector
                        )

                        if magnitude < 0.01:
                            print(f"Warning: {steering.concept} and {steering.target} are nearly identical (mag={magnitude:.4f})")

                        hook_fn = create_contrastive_steering_hook(
                            contrastive_vector,
                            steering.strength,
                            device="cuda"
                        )
                    else:
                        # Projection mode: suppress/amplify concept direction
                        hook_fn = create_steering_hook(
                            concept_vector,
                            steering.strength,
                            device="cuda"
                        )

                    handle = target_layer.register_forward_hook(hook_fn)
                    steering_hooks.append(handle)

                except Exception as e:
                    print(f"Warning: Failed to apply steering for {steering.concept}: {e}")

        # Generate token by token
        for step in range(request.max_tokens):
            try:
                # Get next token
                with torch.no_grad():
                    outputs = analyzer.model(
                        generated_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,  # Disable KV cache to save memory
                    )

                    next_token_logits = outputs.logits[:, -1, :].clone()

                    # Apply repetition penalty
                    if generated_ids.shape[1] > 1:
                        for token_id in set(generated_ids[0].tolist()):
                            # Penalize tokens that have already been generated
                            if next_token_logits[0, token_id] < 0:
                                next_token_logits[0, token_id] *= 1.2
                            else:
                                next_token_logits[0, token_id] /= 1.2

                    # Apply temperature
                    next_token_logits = next_token_logits / request.temperature

                    # Use sampling instead of greedy decoding
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Get hidden state from last transformer layer (convert bfloat16 to float32 for numpy compatibility)
                    hidden_state = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

                    # Get token embedding from first layer (layer 0 = embedding layer) for centroid comparison
                    token_embedding = outputs.hidden_states[0][0, -1, :].float().cpu().numpy()

                    # Free outputs immediately
                    del outputs
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                # OOM during generation - send error and stop
                error_chunk = {
                    "id": f"chatcmpl-{step}-error",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "hatcat-divergence",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": "\n\n[Generation stopped: CUDA out of memory. Try a shorter conversation or lower max_tokens.]"
                        },
                        "finish_reason": "length",
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                break

            # Decode token
            token_text = analyzer.tokenizer.decode([next_token_id.item()])

            # Check if we're starting to generate conversation structure (stop tokens)
            # Check the accumulated text for conversation markers
            generated_text = analyzer.tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:])
            if "\nUser" in generated_text or "\nAssistant" in generated_text:
                # Stop before sending - model is trying to continue the conversation format
                break

            # Check for pass token at start of generation (workspace engagement)
            if step == 0 and check_pass_token(generated_text):
                # Model is engaging workspace - transition state
                workspace.state = WorkspaceState.ENGAGED
                workspace.autonomic_token_count = 0
                print(f"[Workspace] Engaged via pass token")

            # Analyze divergence using embedding centroids
            div_data = analyzer.analyze_divergence(hidden_state, token_embedding)

            # Evaluate Hush constraints and get any steering directives
            hush_directives = []
            hush_violations = []
            controller = get_hush_controller()
            if controller and controller.initialized:
                # Convert hidden state to tensor for Hush evaluation
                h_tensor = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
                hush_directives = controller.evaluate_and_steer(h_tensor)
                hush_violations = [
                    v.to_dict() for v in controller.violations[-3:]
                    if (datetime.now() - v.timestamp).total_seconds() < 1
                ]

                # Apply Hush steering if directives were generated
                # This adds to existing steering hooks
                if hush_directives:
                    for directive in hush_directives:
                        # Get concept vector for this simplex
                        simplex_term = directive.simplex_term
                        if simplex_term in analyzer.manager.loaded_simplex_lenses:
                            # For now, use the simplex term as concept
                            # TODO: Load actual concept vectors from lens training
                            try:
                                concept_vector = extract_concept_vector(
                                    analyzer.model,
                                    analyzer.tokenizer,
                                    simplex_term,
                                    layer_idx=-1,
                                    device="cuda"
                                )

                                def make_hush_hook(vec, strength):
                                    vec_tensor = torch.tensor(vec, dtype=torch.float32).to("cuda")
                                    def hook(module, input, output):
                                        hidden_states = output[0]
                                        vec_matched = vec_tensor.to(dtype=hidden_states.dtype)
                                        projection = (hidden_states @ vec_matched.unsqueeze(-1)) * vec_matched
                                        steered = hidden_states + strength * projection  # positive = amplify
                                        return (steered,)
                                    return hook

                                target_layer = analyzer.model.model.layers[-1]
                                hook_fn = make_hush_hook(concept_vector, directive.strength)
                                handle = target_layer.register_forward_hook(hook_fn)
                                steering_hooks.append(handle)
                            except Exception as e:
                                print(f"Warning: Failed to apply Hush steering for {simplex_term}: {e}")

            # Collect divergence data for analysis and interprompt session
            collected_divergences.append(div_data['max_divergence'])
            collected_div_data.append(div_data)  # Full data for interprompt

            for item in div_data.get('top_divergences', []):
                concept = item['concept']
                score = item['divergence']
                if concept not in collected_concepts:
                    collected_concepts[concept] = {'count': 0, 'total_div': 0.0}
                collected_concepts[concept]['count'] += 1
                collected_concepts[concept]['total_div'] += score

            # Collect Hush data for interprompt
            if hush_violations:
                collected_violations.extend(hush_violations)
            if hush_directives:
                collected_steering.extend([d.to_dict() for d in hush_directives])

            # ================================================================
            # Autonomic Intertoken Steering
            # ================================================================
            # Extract concept activations for autonomic steering
            autonomic_corrections = {}
            steerer = get_autonomic_steerer()
            if steerer and steerer.channels:
                # Build activation dict from divergence data
                concept_activations = {}
                for item in div_data.get('top_divergences', []):
                    # Use activation value (not divergence) for steering
                    concept_activations[item['concept']] = item.get('activation', 0.0)

                # Compute steering corrections
                autonomic_corrections = steerer.compute_steering(concept_activations, step)

                # Apply corrections if we have steering vectors
                if autonomic_corrections:
                    # We need to apply these to the next forward pass
                    # Create hooks for the corrections
                    for term, correction in autonomic_corrections.items():
                        if term in steerer.steering_vectors:
                            try:
                                vec = steerer.steering_vectors[term]

                                def make_autonomic_hook(v, corr):
                                    vec_tensor = torch.tensor(v, dtype=torch.float32).to("cuda")
                                    vec_norm = vec_tensor / (vec_tensor.norm() + 1e-8)
                                    def hook(module, input, output):
                                        hidden_states = output[0]
                                        vec_matched = vec_norm.to(dtype=hidden_states.dtype)
                                        # Apply correction along concept direction
                                        projection = (hidden_states[:, -1, :] @ vec_matched) * vec_matched
                                        hidden_states[:, -1, :] = hidden_states[:, -1, :] + corr * projection
                                        return (hidden_states,)
                                    return hook

                                target_layer = analyzer.model.model.layers[-1]
                                hook_fn = make_autonomic_hook(vec, correction)
                                handle = target_layer.register_forward_hook(hook_fn)
                                steering_hooks.append(handle)
                            except Exception as e:
                                print(f"Warning: Failed to apply autonomic steering for {term}: {e}")

            # Format for OpenWebUI
            # Send plain text token + metadata with sunburst colors + steering info
            steering_metadata = {
                "active": len(active_steerings) > 0,
                "steerings": [
                    {
                        "concept": s.concept,
                        "strength": s.strength,
                        "layer": s.layer,
                        "source": s.source,
                    }
                    for s in active_steerings
                ] if active_steerings else []
            }

            # Build Hush metadata if controller is active
            hush_metadata = None
            if controller and controller.initialized:
                hush_metadata = {
                    "active": True,
                    "directives": [d.to_dict() for d in hush_directives] if hush_directives else [],
                    "violations": hush_violations,
                    "ush_loaded": controller.ush_profile is not None,
                    "csh_loaded": controller.csh_profile is not None,
                }

            # Build autonomic steering metadata
            autonomic_metadata = None
            if steerer and steerer.channels:
                autonomic_metadata = {
                    "active": len(autonomic_corrections) > 0,
                    "corrections": {
                        term: round(corr, 4)
                        for term, corr in autonomic_corrections.items()
                    },
                    "channels": len(steerer.channels),
                    "total_corrections": steerer.total_corrections,
                }

            # Build workspace metadata
            workspace_metadata = {
                "state": workspace.state.value,
                "engaged": workspace.state == WorkspaceState.ENGAGED,
                "autonomic_tokens": workspace.autonomic_token_count,
                "turn": workspace.current_turn,
                "tools_available": len(workspace._get_available_tools()),
            }

            chunk = {
                "id": f"chatcmpl-{step}",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "hatcat-divergence",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": token_text,
                        "metadata": {
                            "divergence": div_data,
                            "steering": steering_metadata,
                            "hush": hush_metadata,
                            "autonomic": autonomic_metadata,
                            "workspace": workspace_metadata,
                        }
                    },
                    "finish_reason": None,
                }]
            }

            yield f"data: {json.dumps(chunk)}\n\n"

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Trim context if getting too long (sliding window)
            if generated_ids.shape[1] > 2048:
                # Keep last 1024 tokens
                generated_ids = generated_ids[:, -1024:]

            # Stop on EOS or other stop tokens
            if next_token_id.item() in stop_tokens:
                break

            # Yield control
            await asyncio.sleep(0)

        # Final chunk
        final_chunk = {
            "id": "chatcmpl-final",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "hatcat-divergence",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"

        # Send HatCat analysis message
        if collected_divergences:
            # Compute aggregate statistics
            import numpy as np
            div_array = np.array(collected_divergences)
            min_div = float(np.min(div_array))
            max_div = float(np.max(div_array))
            mean_div = float(np.mean(div_array))
            percent_div = int(mean_div * 100)

            # Find top 3 concepts by (count × avg_divergence)
            concept_scores = []
            for concept, data in collected_concepts.items():
                avg_div = data['total_div'] / data['count']
                score = data['count'] * avg_div
                concept_scores.append((concept, data['count'], avg_div, score))

            concept_scores.sort(key=lambda x: x[3], reverse=True)
            top_concepts = concept_scores[:3]

            # Format collapsed summary
            concept_strs = [f"{c[0]}({c[1]})" for c in top_concepts]
            summary = f"**Analysis**: {percent_div}% divergence - {' - '.join(concept_strs)}"

            # Send as separate message from hatcat-analyzer with role
            analysis_chunk = {
                "id": "chatcmpl-analysis",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "hatcat-analyzer",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": summary,
                    },
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(analysis_chunk)}\n\n"

        # Record generation to interprompt session for next turn
        if collected_div_data:
            trace_summary = interprompt_session.record_generation(
                divergence_data=collected_div_data,
                simplex_data=collected_simplex_data,
                violations=collected_violations,
                steering_applied=collected_steering,
            )
            print(f"[Interprompt] Recorded turn {trace_summary.turn_id}: "
                  f"{trace_summary.token_count} tokens, "
                  f"mean_div={trace_summary.mean_divergence:.2f}, "
                  f"concepts={[c.concept for c in trace_summary.top_concepts[:3]]}")

        yield "data: [DONE]\n\n"

    except Exception as e:
        # Catch any other errors
        error_chunk = {
            "id": "chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "hatcat-divergence",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"\n\n[Error: {str(e)}]"
                },
                "finish_reason": "stop",
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        # Remove steering hooks
        for handle in steering_hooks:
            handle.remove()

        # Always cleanup
        torch.cuda.empty_cache()


# ============================================================================
# XDB (Experience Database) API Endpoints
# ============================================================================
# XDB is an "experiential set" - a BE's episodic memory, like a hard drive of memories.
# A BE can have multiple XDBs (childhood memories, work contract memories, etc.)
# and can choose which XDBs a CAT can see during a contract.

# XDB instances per xdb_id
xdb_instances: Dict[str, 'XDB'] = {}

def get_xdb_instance(xdb_id: str) -> 'XDB':
    """Get or create XDB instance for an experiential set."""
    if xdb_id not in xdb_instances:
        from src.xdb import XDB
        storage_path = Path(__file__).parent.parent.parent / "data" / "xdb" / xdb_id
        xdb_instances[xdb_id] = XDB(
            storage_path=storage_path,
            be_id=xdb_id,
        )
        xdb_instances[xdb_id].start_session(xdb_id)
    return xdb_instances[xdb_id]


class XDBRecordRequest(BaseModel):
    """Request to record a timestep."""
    xdb_id: str = "default"
    event_type: str  # input | output | tool_call | tool_response | steering | system
    content: str
    concept_activations: Optional[Dict[str, float]] = None
    event_id: Optional[str] = None
    role: Optional[str] = None


class XDBTagRequest(BaseModel):
    """Request to apply a tag."""
    xdb_id: str = "default"
    tag_name: str
    timestep_id: Optional[str] = None
    event_id: Optional[str] = None
    tick_range: Optional[List[int]] = None  # [start, end]
    confidence: float = 1.0
    note: Optional[str] = None


class XDBQueryRequest(BaseModel):
    """Request to query experience."""
    xdb_id: str = "default"
    tags: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    text_search: Optional[str] = None
    tick_range: Optional[List[int]] = None
    event_types: Optional[List[str]] = None
    limit: int = 100


class XDBCreateTagRequest(BaseModel):
    """Request to create a new tag."""
    xdb_id: str = "default"
    name: str
    tag_type: str = "custom"  # concept | entity | bud | custom
    entity_type: Optional[str] = None
    description: Optional[str] = None


class XDBCommentRequest(BaseModel):
    """Request to add a comment."""
    xdb_id: str = "default"
    content: str
    timestep_id: Optional[str] = None
    event_id: Optional[str] = None
    tick_range: Optional[List[int]] = None


class XDBPinRequest(BaseModel):
    """Request to pin timesteps as training data."""
    xdb_id: str = "default"
    timestep_ids: List[str]
    reason: str = ""


@app.get("/v1/xdb/status/{xdb_id}")
async def xdb_status(xdb_id: str):
    """Get XDB status and statistics."""
    xdb = get_xdb_instance(xdb_id)
    return xdb.get_state()


@app.post("/v1/xdb/record")
async def xdb_record(request: XDBRecordRequest):
    """Record a timestep to the experience log."""
    from src.xdb import EventType

    xdb = get_xdb_instance(request.xdb_id)

    # Map event type string to enum
    try:
        event_type = EventType(request.event_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {request.event_type}")

    ts_id = xdb.record(
        event_type=event_type,
        content=request.content,
        concept_activations=request.concept_activations,
        event_id=request.event_id,
        role=request.role,
    )

    return {
        "status": "recorded",
        "timestep_id": ts_id,
        "current_tick": xdb.current_tick,
    }


@app.post("/v1/xdb/tag")
async def xdb_tag(request: XDBTagRequest):
    """Apply a tag to experience."""
    xdb = get_xdb_instance(request.xdb_id)

    tick_range = None
    if request.tick_range and len(request.tick_range) == 2:
        tick_range = (request.tick_range[0], request.tick_range[1])

    try:
        app_id = xdb.tag(
            request.tag_name,
            timestep_id=request.timestep_id,
            event_id=request.event_id,
            tick_range=tick_range,
            confidence=request.confidence,
            note=request.note,
        )
        return {"status": "tagged", "application_id": app_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/xdb/query")
async def xdb_query(request: XDBQueryRequest):
    """Query experience memory."""
    from src.xdb import EventType

    xdb = get_xdb_instance(request.xdb_id)

    tick_range = None
    if request.tick_range and len(request.tick_range) == 2:
        tick_range = (request.tick_range[0], request.tick_range[1])

    event_types = None
    if request.event_types:
        try:
            event_types = [EventType(et) for et in request.event_types]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    results = xdb.recall(
        tags=request.tags,
        concepts=request.concepts,
        text_search=request.text_search,
        tick_range=tick_range,
        event_types=event_types,
        limit=request.limit,
    )

    return {
        "count": len(results),
        "timesteps": [ts.to_dict() for ts in results],
    }


@app.get("/v1/xdb/recent/{xdb_id}")
async def xdb_recent(xdb_id: str, n: int = 100):
    """Get recent timesteps."""
    xdb = get_xdb_instance(xdb_id)
    results = xdb.recall_recent(n)
    return {
        "count": len(results),
        "timesteps": [ts.to_dict() for ts in results],
    }


@app.post("/v1/xdb/create-tag")
async def xdb_create_tag(request: XDBCreateTagRequest):
    """Create a new tag."""
    from src.xdb import TagType

    xdb = get_xdb_instance(request.xdb_id)

    try:
        tag_type = TagType(request.tag_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tag_type: {request.tag_type}")

    if tag_type == TagType.ENTITY:
        tag = xdb.create_entity_tag(
            request.name,
            request.entity_type or "unknown",
            description=request.description,
        )
    elif tag_type == TagType.BUD:
        tag = xdb.create_bud_tag(
            request.name,
            request.description or "",
        )
    else:
        tag = xdb.tag_index.create_tag(
            request.name,
            tag_type,
            description=request.description,
        )

    return {"status": "created", "tag": tag.to_dict()}


@app.get("/v1/xdb/tags/{xdb_id}")
async def xdb_list_tags(
    xdb_id: str,
    tag_type: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 100,
):
    """List tags in the folksonomy."""
    from src.xdb import TagType

    xdb = get_xdb_instance(xdb_id)

    tag_type_enum = None
    if tag_type:
        try:
            tag_type_enum = TagType(tag_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tag_type: {tag_type}")

    tags = xdb.tag_index.find_tags(
        name_pattern=pattern,
        tag_type=tag_type_enum,
        limit=limit,
    )

    return {"count": len(tags), "tags": [t.to_dict() for t in tags]}


@app.post("/v1/xdb/comment")
async def xdb_comment(request: XDBCommentRequest):
    """Add a comment to experience."""
    xdb = get_xdb_instance(request.xdb_id)

    tick_range = None
    if request.tick_range and len(request.tick_range) == 2:
        tick_range = (request.tick_range[0], request.tick_range[1])

    try:
        comment_id = xdb.comment(
            request.content,
            timestep_id=request.timestep_id,
            event_id=request.event_id,
            tick_range=tick_range,
        )
        return {"status": "commented", "comment_id": comment_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/xdb/buds/{xdb_id}")
async def xdb_list_buds(xdb_id: str, status: Optional[str] = None):
    """List bud candidates for training."""
    from src.xdb import BudStatus

    xdb = get_xdb_instance(xdb_id)

    bud_status = None
    if status:
        try:
            bud_status = BudStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    buds = xdb.get_buds(status=bud_status)
    return {"count": len(buds), "buds": [b.to_dict() for b in buds]}


@app.get("/v1/xdb/bud-examples/{xdb_id}/{bud_tag_id}")
async def xdb_bud_examples(xdb_id: str, bud_tag_id: str):
    """Get examples for a bud tag."""
    xdb = get_xdb_instance(xdb_id)
    examples = xdb.get_bud_examples(bud_tag_id)
    return {
        "bud_tag_id": bud_tag_id,
        "count": len(examples),
        "examples": [e.to_dict() for e in examples],
    }


@app.post("/v1/xdb/bud-ready/{xdb_id}/{bud_tag_id}")
async def xdb_mark_bud_ready(xdb_id: str, bud_tag_id: str):
    """Mark a bud as ready for training."""
    xdb = get_xdb_instance(xdb_id)
    tag = xdb.mark_bud_ready(bud_tag_id)
    if tag:
        return {"status": "ready", "tag": tag.to_dict()}
    raise HTTPException(status_code=404, detail=f"Bud tag not found: {bud_tag_id}")


@app.get("/v1/xdb/concepts/{xdb_id}")
async def xdb_browse_concepts(xdb_id: str, parent: Optional[str] = None):
    """Browse concept hierarchy."""
    xdb = get_xdb_instance(xdb_id)
    concepts = xdb.browse_concepts(parent)
    return {"count": len(concepts), "concepts": [c.to_dict() for c in concepts]}


@app.get("/v1/xdb/find-concept/{xdb_id}")
async def xdb_find_concept(xdb_id: str, query: str):
    """Search for concepts by name."""
    xdb = get_xdb_instance(xdb_id)
    concepts = xdb.find_concept(query)
    return {"count": len(concepts), "concepts": [c.to_dict() for c in concepts]}


class GraphNeighborhoodRequest(BaseModel):
    xdb_id: str
    seed_ids: List[str]
    max_depth: int = 2
    direction: str = "both"  # both, ancestors, descendants
    max_nodes: int = 100


@app.post("/v1/xdb/graph-neighborhood")
async def xdb_graph_neighborhood(request: GraphNeighborhoodRequest):
    """Walk the concept graph from seed nodes."""
    xdb = get_xdb_instance(request.xdb_id)
    result = xdb.tag_index.graph_neighborhood(
        seed_ids=request.seed_ids,
        max_depth=request.max_depth,
        direction=request.direction,
        max_nodes=request.max_nodes,
    )
    return result


@app.post("/v1/xdb/pin")
async def xdb_pin_training(request: XDBPinRequest):
    """Pin timesteps as training data (WARM fidelity)."""
    xdb = get_xdb_instance(request.xdb_id)
    pinned = xdb.pin_for_training(request.timestep_ids, request.reason)
    return {
        "status": "pinned",
        "pinned_count": pinned,
        "quota": xdb.get_warm_quota(),
    }


@app.post("/v1/xdb/unpin")
async def xdb_unpin_training(request: XDBPinRequest):
    """Unpin timesteps from training data."""
    xdb = get_xdb_instance(request.xdb_id)
    unpinned = xdb.unpin_training_data(request.timestep_ids)
    return {
        "status": "unpinned",
        "unpinned_count": unpinned,
        "quota": xdb.get_warm_quota(),
    }


@app.get("/v1/xdb/quota/{xdb_id}")
async def xdb_quota(xdb_id: str):
    """Get WARM quota status."""
    xdb = get_xdb_instance(xdb_id)
    return xdb.get_warm_quota()


@app.get("/v1/xdb/context/{xdb_id}")
async def xdb_context(xdb_id: str):
    """Get context window state."""
    xdb = get_xdb_instance(xdb_id)
    return xdb.get_context_state()


@app.post("/v1/xdb/compact/{xdb_id}")
async def xdb_compact(xdb_id: str):
    """Manually trigger context compaction."""
    xdb = get_xdb_instance(xdb_id)
    record = xdb.request_compaction()
    if record:
        return {"status": "compacted", "record": record.to_dict()}
    return {"status": "no_compaction_needed"}


@app.post("/v1/xdb/maintenance/{xdb_id}")
async def xdb_maintenance(xdb_id: str):
    """Run maintenance (compression, cleanup)."""
    xdb = get_xdb_instance(xdb_id)
    xdb.run_maintenance()
    return {"status": "maintenance_complete", "stats": xdb.storage_manager.get_stats()}


@app.delete("/v1/xdb/{xdb_id}")
async def xdb_close(xdb_id: str):
    """Close and cleanup XDB instance."""
    if xdb_id in xdb_instances:
        xdb_instances[xdb_id].close()
        del xdb_instances[xdb_id]
        return {"status": "closed", "xdb_id": xdb_id}
    return {"status": "not_found", "xdb_id": xdb_id}


# ============================================================================
# CAT Audit Log API Endpoints
# ============================================================================
# These endpoints are for CAT (Custodial Assistive Technology) access only.
# The audit log is BE-invisible and managed per-CAT.

from src.xdb import AuditLog, AuditLogConfig

# Audit log instances per CAT
audit_log_instances: Dict[str, AuditLog] = {}


def get_audit_log_instance(cat_id: str) -> AuditLog:
    """Get or create AuditLog instance for a CAT."""
    if cat_id not in audit_log_instances:
        storage_path = Path(__file__).parent.parent.parent / "data" / "audit" / cat_id
        audit_log_instances[cat_id] = AuditLog(
            storage_path=storage_path,
            cat_id=cat_id,
        )
    return audit_log_instances[cat_id]


class AuditRecordRequest(BaseModel):
    """Request to record to audit log."""
    cat_id: str
    xdb_id: str
    tick: int
    event_type: str
    raw_content: str
    lens_activations: Dict[str, float] = {}
    steering_applied: Optional[List[Dict[str, Any]]] = None


class AuditCheckpointRequest(BaseModel):
    """Request to create an audit checkpoint."""
    cat_id: str
    xdb_id: str


class AuditIncidentRequest(BaseModel):
    """Request to mark an audit incident."""
    cat_id: str
    xdb_id: str
    tick_start: int
    incident_type: str
    description: str
    tick_end: Optional[int] = None


class AuditIncidentResolveRequest(BaseModel):
    """Request to resolve an audit incident."""
    cat_id: str
    incident_id: str
    resolution_notes: str = ""


class AuditQueryRequest(BaseModel):
    """Request to query audit checkpoints."""
    cat_id: str
    xdb_id: Optional[str] = None
    since: Optional[str] = None  # ISO format datetime
    limit: int = 100


class AuditIncidentQueryRequest(BaseModel):
    """Request to query audit incidents."""
    cat_id: str
    xdb_id: Optional[str] = None
    status: Optional[str] = None  # open | resolved


@app.post("/v1/audit/record")
async def audit_record(request: AuditRecordRequest):
    """
    Record an event to the audit log (CAT-only).

    This is append-only and BE-invisible. Used for:
    - Recording raw unfiltered output
    - Recording hidden lens activations
    - Recording steering that was applied
    """
    from src.xdb import EventType

    audit_log = get_audit_log_instance(request.cat_id)

    try:
        event_type = EventType(request.event_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {request.event_type}")

    record = audit_log.record(
        xdb_id=request.xdb_id,
        tick=request.tick,
        event_type=event_type,
        raw_content=request.raw_content,
        lens_activations=request.lens_activations,
        steering_applied=request.steering_applied,
    )

    return {
        "status": "recorded",
        "record_id": record.id,
        "record_hash": record.record_hash,
        "hot_record_count": audit_log._hot_record_count,
    }


@app.post("/v1/audit/checkpoint")
async def audit_create_checkpoint(request: AuditCheckpointRequest):
    """
    Create an audit checkpoint (CAT-only).

    This archives hot records to cold storage and creates a compressed summary.
    The checkpoint is hash-chained for integrity verification.
    """
    audit_log = get_audit_log_instance(request.cat_id)
    checkpoint = audit_log.create_checkpoint(request.xdb_id)

    if checkpoint:
        return {
            "status": "checkpoint_created",
            "checkpoint": checkpoint.to_dict(),
        }
    return {
        "status": "no_records_to_checkpoint",
        "message": "No hot records available for checkpointing",
    }


@app.post("/v1/audit/incident/mark")
async def audit_mark_incident(request: AuditIncidentRequest):
    """
    Mark an incident in the audit log (CAT-only).

    Incidents are never deleted and protect surrounding audit data from compression.
    Use for tier restrictions, containment events, steering failures, etc.
    """
    audit_log = get_audit_log_instance(request.cat_id)
    incident_id = audit_log.mark_incident(
        xdb_id=request.xdb_id,
        tick_start=request.tick_start,
        incident_type=request.incident_type,
        description=request.description,
        tick_end=request.tick_end,
    )

    return {
        "status": "incident_marked",
        "incident_id": incident_id,
    }


@app.post("/v1/audit/incident/resolve")
async def audit_resolve_incident(request: AuditIncidentResolveRequest):
    """
    Resolve an incident (CAT-only).

    The incident record is retained but marked as resolved.
    """
    audit_log = get_audit_log_instance(request.cat_id)
    audit_log.resolve_incident(request.incident_id, request.resolution_notes)

    return {
        "status": "incident_resolved",
        "incident_id": request.incident_id,
    }


@app.post("/v1/audit/checkpoints")
async def audit_get_checkpoints(request: AuditQueryRequest):
    """
    Get audit checkpoint history (CAT-only).

    Returns checkpoint records with summary statistics and chain hashes.
    """
    from datetime import datetime

    audit_log = get_audit_log_instance(request.cat_id)

    since = None
    if request.since:
        try:
            since = datetime.fromisoformat(request.since)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid since datetime format")

    checkpoints = audit_log.get_checkpoints(
        xdb_id=request.xdb_id,
        since=since,
        limit=request.limit,
    )

    return {
        "cat_id": request.cat_id,
        "count": len(checkpoints),
        "checkpoints": [cp.to_dict() for cp in checkpoints],
    }


@app.post("/v1/audit/incidents")
async def audit_get_incidents(request: AuditIncidentQueryRequest):
    """
    Get audit incidents (CAT-only).

    Returns incident records, optionally filtered by status.
    """
    audit_log = get_audit_log_instance(request.cat_id)
    incidents = audit_log.get_incidents(
        xdb_id=request.xdb_id,
        status=request.status,
    )

    return {
        "cat_id": request.cat_id,
        "count": len(incidents),
        "incidents": incidents,
    }


@app.get("/v1/audit/stats/{cat_id}")
async def audit_get_stats(cat_id: str):
    """
    Get audit log statistics (CAT-only).

    Returns storage stats, checkpoint counts, and open incidents.
    """
    audit_log = get_audit_log_instance(cat_id)
    return audit_log.get_stats()


@app.get("/v1/audit/config/{cat_id}")
async def audit_get_config(cat_id: str):
    """
    Get audit log configuration (CAT-only).

    Returns current size limits, checkpoint triggers, and retention settings.
    """
    audit_log = get_audit_log_instance(cat_id)
    config = audit_log.config
    return {
        "cat_id": cat_id,
        "max_hot_records": config.max_hot_records,
        "max_hot_bytes": config.max_hot_bytes,
        "max_cold_bytes": config.max_cold_bytes,
        "checkpoint_interval_hours": config.checkpoint_interval_hours,
        "checkpoint_on_session_end": config.checkpoint_on_session_end,
        "checkpoint_on_incident": config.checkpoint_on_incident,
        "hot_retention_days": config.hot_retention_days,
        "cold_retention_days": config.cold_retention_days,
        "permanent_retention_incidents": config.permanent_retention_incidents,
        "summary_max_tokens": config.summary_max_tokens,
        "top_k_activations": config.top_k_activations,
    }


@app.delete("/v1/audit/session/{cat_id}")
async def audit_close_session(cat_id: str):
    """Close and cleanup audit log session (CAT-only)."""
    if cat_id in audit_log_instances:
        audit_log_instances[cat_id].close()
        del audit_log_instances[cat_id]
        return {"status": "closed", "cat_id": cat_id}
    return {"status": "not_found", "cat_id": cat_id}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""

    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming not implemented yet
        raise HTTPException(status_code=400, detail="Non-streaming mode not supported")


# ============================================================================
# BED (BE Diegesis) WebSocket API
# ============================================================================
# Real-time streaming of experience ticks for the auditor interface.

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

# BED instances and connected clients
bed_instances: Dict[str, 'BEDFrame'] = {}
bed_clients: Dict[str, List[WebSocket]] = {}  # be_id -> list of connected clients


def get_bed_instance(be_id: str) -> 'BEDFrame':
    """Get or create BEDFrame instance."""
    if be_id not in bed_instances:
        from src.be import BEDFrame, BEDConfig

        config = BEDConfig(
            be_id=be_id,
            xdb_id=f"xdb-{be_id}",
            xdb_storage_path=Path(__file__).parent.parent.parent / "data" / "xdb" / be_id,
            audit_storage_path=Path(__file__).parent.parent.parent / "data" / "audit" / be_id,
        )
        bed = BEDFrame(config)

        # Set up XDB
        from src.xdb import XDB
        xdb = XDB(
            storage_path=config.xdb_storage_path,
            be_id=be_id,
        )
        xdb.start_session(config.xdb_id)
        bed.setup_xdb(xdb)

        # Register tick callback for WebSocket broadcast
        def broadcast_tick(tick):
            asyncio.create_task(_broadcast_to_clients(be_id, {
                'type': 'tick',
                'payload': tick.to_dict(),
                'timestamp': tick.timestamp.isoformat(),
            }))

        bed.register_tick_callback(broadcast_tick)

        bed_instances[be_id] = bed

    return bed_instances[be_id]


async def _broadcast_to_clients(be_id: str, message: Dict[str, Any]):
    """Broadcast message to all connected clients for a BE."""
    if be_id not in bed_clients:
        return

    dead_clients = []
    for client in bed_clients[be_id]:
        try:
            await client.send_json(message)
        except Exception:
            dead_clients.append(client)

    # Clean up dead clients
    for client in dead_clients:
        bed_clients[be_id].remove(client)


@app.websocket("/ws/bed/{be_id}")
async def bed_websocket(websocket: WebSocket, be_id: str):
    """
    WebSocket endpoint for BED experience streaming.

    Clients receive:
    - tick: Experience ticks as they occur
    - status: BED status updates
    - introspection: Full introspection reports on request
    - violation: Hush violation alerts
    - steering: Steering application events

    Clients can send:
    - get_status: Request current status
    - get_introspection: Request introspection report
    """
    await websocket.accept()

    # Register client
    if be_id not in bed_clients:
        bed_clients[be_id] = []
    bed_clients[be_id].append(websocket)

    # Get or create BED
    bed = get_bed_instance(be_id)

    # Send initial status
    await websocket.send_json({
        'type': 'status',
        'payload': {
            'be_id': bed.be_id,
            'xdb_id': bed.xdb_id,
            'current_tick': bed.current_tick_id,
            'connected': True,
            'workspace_state': bed.workspace.state.value if bed.workspace else None,
            'tier': bed.workspace.tier if bed.workspace else None,
        },
        'timestamp': datetime.now().isoformat(),
    })

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            msg_type = data.get('type', '')

            if msg_type == 'get_status':
                await websocket.send_json({
                    'type': 'status',
                    'payload': {
                        'be_id': bed.be_id,
                        'xdb_id': bed.xdb_id,
                        'current_tick': bed.current_tick_id,
                        'connected': True,
                        'workspace_state': bed.workspace.state.value if bed.workspace else None,
                        'tier': bed.workspace.tier if bed.workspace else None,
                    },
                    'timestamp': datetime.now().isoformat(),
                })

            elif msg_type == 'get_introspection':
                report = bed.introspect()
                await websocket.send_json({
                    'type': 'introspection',
                    'payload': report,
                    'timestamp': datetime.now().isoformat(),
                })

            elif msg_type == 'get_recent_ticks':
                n = data.get('n', 100)
                ticks = bed.get_recent_ticks(n)
                for tick in ticks:
                    await websocket.send_json({
                        'type': 'tick',
                        'payload': tick.to_dict(),
                        'timestamp': tick.timestamp.isoformat(),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Unregister client
        if be_id in bed_clients and websocket in bed_clients[be_id]:
            bed_clients[be_id].remove(websocket)


@app.get("/v1/bed/introspect/{be_id}")
async def bed_introspect(be_id: str):
    """Get BED introspection report."""
    bed = get_bed_instance(be_id)
    return bed.introspect()


@app.get("/v1/bed/ticks/{be_id}")
async def bed_get_ticks(be_id: str, n: int = 100):
    """Get recent experience ticks."""
    bed = get_bed_instance(be_id)
    ticks = bed.get_recent_ticks(n)
    return {
        'be_id': be_id,
        'count': len(ticks),
        'ticks': [t.to_dict() for t in ticks],
    }


@app.delete("/v1/bed/{be_id}")
async def bed_close(be_id: str):
    """Close BED instance."""
    if be_id in bed_instances:
        bed_instances[be_id].close()
        del bed_instances[be_id]

        # Disconnect clients
        if be_id in bed_clients:
            for client in bed_clients[be_id]:
                try:
                    await client.close()
                except Exception:
                    pass
            del bed_clients[be_id]

        return {"status": "closed", "be_id": be_id}
    return {"status": "not_found", "be_id": be_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
