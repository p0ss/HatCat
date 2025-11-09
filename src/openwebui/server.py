#!/usr/bin/env python3
"""
FastAPI server for HatCat divergence visualization in OpenWebUI.

This provides an OpenAI-compatible API with divergence metadata.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import torch
import numpy as np
from pathlib import Path
import asyncio
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


class DivergenceAnalyzer:
    """Singleton divergence analyzer."""

    def __init__(self):
        self.manager = None
        self.model = None
        self.tokenizer = None
        self.color_mapper = None
        self.initialized = False

    async def initialize(self):
        """Load models and probes."""
        if self.initialized:
            return

        print("🎩 Initializing HatCat divergence analyzer...")

        from src.monitoring.dynamic_probe_manager import DynamicProbeManager
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load probe manager using probe pack
        self.manager = DynamicProbeManager(
            probe_pack_id="gemma-3-4b-pt_sumo-wordnet-v1",
            base_layers=[0],
            use_activation_probes=True,
            use_text_probes=True,
            keep_top_k=100,
        )

        # Load model
        model_name = "google/gemma-3-4b-pt"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda",
        )
        self.model.eval()

        # Load color mapper
        self.color_mapper = get_color_mapper()

        self.initialized = True
        print(f"✓ Loaded {len(self.manager.loaded_activation_probes)} activation probes")
        print(f"✓ Loaded {len(self.manager.loaded_text_probes)} text probes")
        print(f"✓ Loaded sunburst color mapping for {len(self.color_mapper.positions)} concepts")

    def analyze_divergence(self, hidden_state: np.ndarray, token_text: str) -> Dict[str, Any]:
        """Analyze divergence for a token."""

        # Run activation probes
        activation_scores = {}
        for concept_key, probe in self.manager.loaded_activation_probes.items():
            with torch.no_grad():
                h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
                prob = probe(h).item()
                if prob > 0.5:  # Only high confidence
                    activation_scores[concept_key[0]] = prob

        # Run text probes
        text_scores = {}
        for concept_key, text_probe in self.manager.loaded_text_probes.items():
            try:
                prob = text_probe.pipeline.predict_proba([token_text])[0, 1]
                if prob > 0.5:  # Only high confidence
                    text_scores[concept_key[0]] = prob
            except:
                pass

        # Calculate divergences
        all_concepts = set(activation_scores.keys()) | set(text_scores.keys())
        divergences = []
        concepts_with_data = []  # For color mapping

        for concept in all_concepts:
            act_prob = activation_scores.get(concept, 0.0)
            txt_prob = text_scores.get(concept, 0.0)
            div = abs(act_prob - txt_prob)
            if div > 0.2:  # Only significant divergences
                divergences.append({
                    'concept': concept,
                    'activation': round(act_prob, 3),
                    'text': round(txt_prob, 3),
                    'divergence': round(div, 3),
                })
                # For color mapping: (concept, activation, divergence)
                concepts_with_data.append((concept, act_prob, div))

        divergences.sort(key=lambda x: x['divergence'], reverse=True)

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
            'max_divergence': divergences[0]['divergence'] if divergences else 0.0,
            'top_divergences': divergences[:3],
            'activation_detections': [(k, round(v, 3)) for k, v in sorted(activation_scores.items(), key=lambda x: -x[1])[:3]],
            'text_detections': [(k, round(v, 3)) for k, v in sorted(text_scores.items(), key=lambda x: -x[1])[:3]],
            'token_color': token_color,
            'palette': palette,
        }


# Global analyzer instance
analyzer = DivergenceAnalyzer()


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
        "activation_probes": len(analyzer.manager.loaded_activation_probes) if analyzer.manager else 0,
        "text_probes": len(analyzer.manager.loaded_text_probes) if analyzer.manager else 0,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    packs = registry.get_pack_summary()

    # Build model list from available probe packs
    models = []
    for pack in packs:
        model_id = f"{pack['model_id']}_{pack['concept_pack_id']}"
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1234567890,
            "owned_by": "hatcat",
            "probe_pack_id": pack['probe_pack_id'],
            "concept_pack_id": pack['concept_pack_id'],
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


@app.get("/v1/probe-packs")
async def list_probe_packs():
    """List available probe packs."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    return {
        "probe_packs": registry.get_pack_summary()
    }


@app.get("/v1/probe-packs/{pack_id}")
async def get_probe_pack(pack_id: str):
    """Get details for a specific probe pack."""
    from src.registry import ProbePackRegistry

    registry = ProbePackRegistry()
    pack = registry.get_pack(pack_id)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Probe pack not found: {pack_id}")

    return pack.pack_json


async def generate_stream(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming response with divergence coloring."""

    try:
        if not analyzer.initialized:
            await analyzer.initialize()

        # Build prompt from messages (only recent context to save memory)
        max_context_messages = 10  # Limit context to prevent OOM
        recent_messages = request.messages[-max_context_messages:]

        prompt_parts = []
        for msg in recent_messages:
            prompt_parts.append(f"{msg.role}: {msg.content}")
        prompt = "\n".join(prompt_parts) + "\nassistant: "

        # Tokenize with truncation
        inputs = analyzer.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Limit input length
        ).to("cuda")
        generated_ids = inputs.input_ids

        # Clear CUDA cache before generation
        torch.cuda.empty_cache()

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

                    next_token_logits = outputs.logits[:, -1, :] / request.temperature
                    next_token_id = torch.argmax(next_token_logits, dim=-1)

                    # Get hidden state
                    hidden_state = outputs.hidden_states[0][0, -1, :].cpu().numpy()

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

            # Analyze divergence
            div_data = analyzer.analyze_divergence(hidden_state, token_text)

            # Format for OpenWebUI
            # Send plain text token + metadata with sunburst colors
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
                            "color": div_data['token_color'],
                            "palette": div_data['palette'],
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

            # Stop on EOS
            if next_token_id.item() == analyzer.tokenizer.eos_token_id:
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
        # Always cleanup
        torch.cuda.empty_cache()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
    )
