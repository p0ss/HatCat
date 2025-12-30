# OpenWebUI Fork Progress

## Completed

### 1. Repository Setup ✅
- Cloned OpenWebUI to `/home/poss/Documents/Code/hatcat-ui`
- Created feature branch: `feature/hatcat-divergence-viz`
- Added upstream remote

### 2. Streaming API Modifications ✅
**File**: `src/lib/apis/streaming/index.ts`

**Changes:**
1. Added `metadata` field to `TextStreamUpdate` type (lines 14-19)
2. Extract metadata from delta in `openAIStreamToIterator` (lines 92-115)
   - Extracts `delta.metadata` from chunks
   - **Security**: Validates color values (must be #RRGGBB)
   - Validates palette array (max 10 colors)
   - Sanitizes to prevent injection attacks
3. Preserve metadata through chunking in `streamLargeDeltasAsRandomChunks` (lines 135-138)

**What this does:**
- When HatCat server sends chunks with `delta.metadata`, the streaming system captures and validates it
- Metadata includes: divergence scores, color, palette, steering info
- Validated colors prevent CSS injection
- All downstream components now have access to this safe metadata

### 3. Message Type Extensions ✅
**File**: `src/lib/components/chat/Messages/ResponseMessage.svelte`

**Changes:**
1. Added `tokenMetadata` field to `MessageType` interface (lines 115-124)
   - Array of `{token: string, metadata?: {...}}`
   - Stores metadata alongside content for rendering

**What this does:**
- Messages can now store per-token metadata
- Enables token-level coloring without modifying content string
- Optional field - backward compatible with existing messages

### 4. Token Metadata Collection ✅
**File**: `src/lib/components/chat/Chat.svelte`

**Changes:**
1. Extract metadata from streaming updates (line 2225)
2. Initialize tokenMetadata array before streaming loop (lines 2224-2227)
3. Store token+metadata pairs as they arrive (lines 2240-2243)

**What this does:**
- Accumulates token metadata parallel to content string
- Each token is stored with its divergence color and data
- Ready for rendering with colored backgrounds

### 5. Analysis Message Generation ✅
**File**: `/home/poss/Documents/Code/HatCat/src/openwebui/server.py`

**Changes:**
1. Collect divergence scores during generation (lines 442-444, 530-537)
2. Compute aggregate statistics after generation (lines 608-630)
3. Send analysis message with `model: "hatcat-analyzer"` (lines 632-647)

**What this does:**
- After each model response, HatCat sends follow-up analysis message
- Format: `69% divergence - Process(10) - Collection(7) - PhysicalSystem(7)`
- Appears as separate agent message (different model ID)
- **TESTED** ✅ - Analysis messages successfully sent via SSE

## Current Status

**What's Working:**
- ✅ Analysis messages are generated server-side after each response
- ✅ Messages include aggregate statistics: `69% divergence - Process(10) - Collection(7) - PhysicalSystem(7)`
- ✅ Analysis sent as separate SSE chunk with `model: "hatcat-analyzer"`
- ✅ Streaming infrastructure extracts and validates metadata
- ✅ Token metadata collection in Chat component

**Blocked:**
- ❌ OpenWebUI dev server requires Node 20+ (currently have Node 18.19.1)
- Need to upgrade Node to test frontend integration

## Next Steps

### 6. Test Analysis Message Rendering (BLOCKED: Need Node 20+)
**What should happen:**
- When analysis message arrives with `model: "hatcat-analyzer"`, OpenWebUI should create a separate message
- Analysis summary should appear after model response
- Should display as coming from "hatcat-analyzer" model

**How to test:**
1. Upgrade to Node 20+: `nvm install 20 && nvm use 20`
2. Install dependencies: `cd /home/poss/Documents/Code/hatcat-ui && npm install --legacy-peer-deps`
3. Start dev server: `npm run dev`
4. Configure HatCat as external model in OpenWebUI
5. Send a message and verify analysis message appears

### 7. Create Token Rendering Component
Once we find where tokens are rendered, create a component that:
- Checks if metadata exists on each token
- Applies background color from `metadata.color`
- Calculates contrasting text color
- Adds tooltip with divergence details

### 5. Test End-to-End
- Build frontend: `npm run dev`
- Start HatCat server
- Configure HatCat as external model in OpenWebUI
- Test chat with colored tokens

### 6. Polish and Document
- Add settings toggle
- Create installation guide
- Prepare upstream PR

## Architecture

```
HatCat Server (localhost:8765)
  ↓
SSE Stream with metadata in delta
  ↓
streaming/index.ts [DONE]
  - Extracts metadata
  - Passes through stream
  ↓
??? (Need to find)
  - Accumulates tokens
  - Builds message
  ↓
ContentRenderer/Markdown [TODO]
  - Apply colors based on metadata
  - Render tooltips
  ↓
User sees colored tokens
```

## Commands

```bash
# Development
cd /home/poss/Documents/Code/hatcat-ui
npm install
npm run dev

# Start HatCat server
cd /home/poss/Documents/Code/HatCat
poetry run python src/openwebui/server.py

# Commit changes
git add src/lib/apis/streaming/index.ts
git commit -m "Add support for token metadata in streaming"
```

## Files Modified

1. `/home/poss/Documents/Code/hatcat-ui/src/lib/apis/streaming/index.ts`
   - Added metadata support to streaming infrastructure
   - ~10 lines changed

## Files to Create

1. Token renderer component (if needed)
2. Tooltip component for divergence details
3. Settings toggle for visualization

## Estimated Remaining Time

- Find rendering location: 30 min
- Create token component: 1 hour
- Test and debug: 1 hour
- Polish: 30 min

**Total**: ~3 hours
