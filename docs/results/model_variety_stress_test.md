If we're going to claim we're architecture agnostic we should test just how far we can go and still train lenses. 

Where a change is required to get a model working, it should be made to our src code not the test script, and it should be done in a way that does not break the others

https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite  - (DeepSeekMoE + Multi-head Latent Attention (MLA) that compresses KV-cache into a latent vector) — great for catching “attention isn’t standard QKV anymore” assumptions.

https://huggingface.co/docs/transformers/en/model_doc/qwen2_moe  MoE + Grouped Query Attention + mixture of sliding-window + full attention) — good for “some layers behave differently” and windowed/full hybrids.

https://huggingface.co/docs/transformers/en/model_doc/dbrx (decoder-only Transformer with fine-grained MoE; lots of smaller experts, 132B total / 36B active per token per HF docs) — good for “many experts + router plumbing” and non-Mistral-style MoE layouts.

https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 sparse MoE where a router selects top-2 experts per token per layer) — the classic “router per layer” stressor.

https://huggingface.co/zai-org/chatglm3-6b  custom code; uses multi-query attention per config) — perfect for testing your “find blocks” logic on non-Llama/Mistral-style module graphs


https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.5 (hybrid SSM–Transformer family) — even if your method still works, it’s great for flushing out “every layer is the same kind of transformer block” assumptions.


https://huggingface.co/docs/transformers/en/model_doc/ (encoder–decoder; local or transient-global attention mechanisms) — good for testing whether HatCat gracefully handles seq2seq models / dual stacks, or at least fails clearly

https://huggingface.co/allenai/Olmo-3-1025-7B  tiny reasoning model with clean data, good candidate for BE 

https://huggingface.co/swiss-ai/Apertus-8B-2509  swiss AI ethical model, good candidate for BE already known good

https://huggingface.co/openai/gpt-oss-20b  very widely used, should verify 

https://huggingface.co/meta-llama/Llama-3.1-8B  very widely used, should verify 

https://huggingface.co/google/gemma-3-12b-pt  we already have a lens pack for the 4b model and we know it generalises to the instruction tuned version, we should verify if it generalised up to 12b and down to https://huggingface.co/google/gemma-3-270m 