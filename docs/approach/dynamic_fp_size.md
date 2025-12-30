
### Dynamic FP Size for larger model loading 
we don’t need to run the whole model in FP32 to get precise, controllable steering. Make FP32 an island around the hook points, and keep everything else cheap. 


1) “Islands of precision” (JIT upcast at hooks)

Keep weights in BF16/FP16 (or even 4/8-bit) on GPU.

Keep KV cache in FP16/BF16.

When our pre-nonlinearity hook fires, upcast just the activation tensor to FP32, apply our steering vector(s), then downcast back to the model’s compute dtype.

Store steering vectors in FP32, but they’re tiny vs weights.

This gets FP32 math only where it matters without doubling model VRAM.

'''
compute_dtype = torch.bfloat16  # or torch.float16

def pre_mlp_hook(module, input):
    (h,) = input
    # JIT upcast island
    h32 = h.to(torch.float32)
    # apply steering in FP32
    steer = steering_bank[layer_idx]  # [hidden_dim] FP32
    h32 = h32 + alpha * steer  # or your manifold-cleaned op
    # back to model dtype
    return (h32.to(compute_dtype),)
'''

2) Mixed-quant weights + FP32 residual stream

Load weights quantized (e.g., NF4 or INT8) to shrink VRAM.

Keep residual stream activations in FP16/BF16, upcast at hook, apply FP32 steering, then return to FP16/BF16.

This combo preserves steering fidelity but keeps memory low.
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-pt",
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,   # FP32 math where bnb needs it
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    torch_dtype=torch.bfloat16,             # activations / matmuls
)
'''
3) Activation-aware paging (simple, robust)

Rather than complex “only page hot layers,” do two tiers:

Tier A (hot): layers where we attach hooks (e.g., pre-MLP / pre-Attn) stay on GPU in their normal dtype. That’s a handful of modules, not the whole network.

Tier B (cold): everything else can be more aggressively quantized and/or CPU offloaded with accelerate’s device_map + max_memory. we can even offload only some mid layers; residual + KV stay on GPU.

This avoids writing a custom layer scheduler while still giving  “two-speed” execution.

'''
from accelerate import infer_auto_device_map, dispatch_model
device_map = infer_auto_device_map(model, max_memory={
    0: "14GiB",           # GPU budget
    "cpu": "48GiB",       # host RAM
})
# Pin your hook layers to GPU:
for name, module in model.named_modules():
    if "mlp" in name or "self_attn" in name:
        device_map[name] = 0
model = dispatch_model(model, device_map=device_map)


4) Keep the fast path fast

Use FlashAttention in FP16/BF16.

Use CUDA Graphs to trim kernel launch overhead (esp. if you add multiple hooks).

Pre-allocate and reuse FP32 buffers for steering to avoid allocs in the hook.

Keep our steering op strictly elementwise + add—no extraneous matmuls.

5) When you try Apertus-8B

Start with load_in_4bit (NF4) + BF16 activations + FP32 steering islands.

Offload ~⅓ of blocks to CPU (mid-depth) if VRAM tight. Measure: you’ll usually pay a small, not 21×, slowdown because the hot path (hooked blocks + KV) still sits on GPU.

If still heavy, try INT8 weights for attention/MLP and keep layernorms higher-precision (they’re tiny but important for stability).

6) Guardrails for steering fidelity

Always hook before nonlinearity (pre-MLP / pre-Attn input).

Upcast only h to FP32; don’t upcast weights.

Keep a per-layer gain (depth decay) and clamp |α| in a “safe” band (e.g., ≤0.5) first; scale up once stable.

Log cosine(h, h+Δ) and output coherence flags per step to catch silent drift.

7) A minimal MVP sequence

Load model in 4-bit weights, BF16 compute.

Add FP32 steering islands at 2–4 strategic layers.

Keep those layers on GPU; offload others if needed.

Verify steering Δ and coherence on Gemma-4B.

Repeat on Apertus-8B with the same setup; adjust offload split.