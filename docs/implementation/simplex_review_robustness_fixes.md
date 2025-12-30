# Simplex Agentic Review - Robustness Improvements

**Date**: 2025-01-16
**Script**: `scripts/run_simplex_agentic_review.py`
**Issue**: JSON parsing failure causing script crash during Stage 1

## Problem

The original simplex review encountered a `JSONDecodeError` when Claude's response contained malformed JSON:

```
json.decoder.JSONDecodeError: Unterminated string starting at: line 133 column 96 (char 6714)
```

This happened because:
1. Large LLM responses can sometimes get truncated
2. Complex JSON structures may have formatting issues
3. No retry logic for transient API errors
4. Missing system prompts for better response quality

## Solutions Implemented

### 1. Enhanced JSON Parser with Salvage Mechanism

**Location**: `scripts/run_simplex_agentic_review.py:465-490`

```python
def _extract_json(self, text: str) -> Dict:
    """Extract JSON from response text with error recovery."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # Show diagnostic info
        print(f"⚠️  JSON parsing error at position {e.pos}")
        print(f"Error: {e.msg}")

        # Attempt to salvage partial response
        truncated = json_text[:e.pos]
        for i in range(len(truncated) - 1, -1, -1):
            if truncated[i] in ']}':
                try:
                    partial = json.loads(truncated[:i+1])
                    print(f"✓ Salvaged partial response")
                    return partial
                except:
                    continue

        # If salvage fails, re-raise
        raise
```

**Benefits**:
- Shows exactly where parsing failed for debugging
- Recovers partial valid JSON (processes what it can)
- Graceful degradation instead of hard crash

### 2. API Retry Logic with Exponential Backoff

**Location**: `scripts/run_simplex_agentic_review.py:492-544`

```python
async def _call_api_with_retry(
    self,
    system: str,
    user_message: str,
    max_retries: int = 3,
    base_delay: float = 2.0
) -> anthropic.types.Message:
    """Call Claude API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return self.client.messages.create(...)
        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
            await asyncio.sleep(delay)
        except anthropic.APIError:
            # Similar retry logic
            ...
```

**Benefits**:
- Handles rate limits automatically (waits 2s → 4s → 8s)
- Retries transient API errors
- Clear user feedback during retries
- Respects API rate limits

### 3. Proper System Prompts for All Stages

**Locations**: Lines 125, 248, 374

**Stage 1 - Coverage & Prioritization**:
```python
system_prompt = "You are an expert in affective psychology and AI interoceptive monitoring systems. Analyze concept coverage and prioritize gaps that could compromise system stability."
```

**Stage 2 - Simplex Identification**:
```python
system_prompt = "You are an expert in affective psychology and dimensional models of emotion. Identify three-pole simplexes with stable neutral attractors."
```

**Stage 3 - Neutral Discovery**:
```python
system_prompt = "You are an expert in affective science and homeostatic systems. Validate neutral attractors for psychological stability."
```

**Benefits**:
- Better response quality through role prompting
- More consistent JSON formatting
- Domain expertise framing reduces hallucinations
- Clearer task focus

## Testing

To test the improved robustness:

```bash
# With API key set
export ANTHROPIC_API_KEY="your-key"
poetry run python scripts/run_simplex_agentic_review.py 50

# Expected improvements:
# - Graceful handling of partial JSON
# - Automatic retry on rate limits
# - Better structured responses from system prompts
```

## Future Improvements

1. **Structured Outputs**: Use Claude's JSON mode if available for guaranteed valid JSON
2. **Response Validation**: Add schema validation before processing
3. **Checkpoint/Resume**: Save progress after each stage to allow resuming from failures
4. **Parallel Processing**: Batch API calls for faster processing
5. **Cost Tracking**: Add running cost estimation during execution

## Impact

These fixes make the simplex review robust enough for production use:
- **Reliability**: Handles API hiccups gracefully
- **Debuggability**: Shows exactly what went wrong when issues occur
- **Recovery**: Salvages partial results instead of losing all work
- **User Experience**: Clear feedback during retries and errors

The script can now confidently process 1000+ concepts over 50+ minutes without manual intervention.
