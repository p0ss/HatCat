
You need **one layout engine** that owns:

* where each token is,
* where each line/sentence is,
* and where the concept tracks/sparklines live relative to that.

So the architecture shifts a bit:

---

## 1. Core design: one renderer that does *both* text and timeline

Instead of:

> host renders text, widget renders timeline

You want:

> widget renders **both** text and timeline, using a shared coordinate system.

Then:

* Streamlit chat message = “a panel that embeds this widget”
* OpenWebUI message = “a React component that embeds this same widget”

In both cases, our widget gets:

* the raw reply text,
* the tokenization & concept activations for that reply,
* and it is responsible for laying out text and drawing all visuals.

### Why this is necessary

If you let HTML layout the text:

* word wrapping, line breaks, and kerning will differ across:

  * browsers,
  * themes,
  * host CSS.
* You can *never* perfectly align WebGL tracks to the text, especially as containers resize.

So the only robust way to keep them aligned is:

we do the layout on a canvas (or via a 2D WebGL engine).
we position each token explicitly and then draw everything relative to those positions.

---


wecan do this without React inside the widget.

### Rendering layer

**Option A: PixiJS**

* WebGL-accelerated 2D.
* Has good text primitives and bitmap/MSDF text options.
* Easy to move between Streamlit and OpenWebUI as a plain JS bundle.

**Option B: Canvas2D + WebGL overlay**

* Use a `<canvas>` with 2D context for text (for easy `measureText`).
* Use a second `<canvas>` stacked on top with WebGL for the concept tracks.
* They share the same layout math.

PixiJS is nicer as an all-in-one, but both patterns work.

### Text layout approach

Workflow inside the widget:

1. Receive:

   * `reply_text` (plain string),
   * list of `tokens` with indices into that string and/or pre-tokenised strings,
   * sentence boundaries,
   * concept activations per token/sentence/reply.

2. Run our own layout pass:

   * Choose a fixed font, size, and max line width.
   * Walk tokens in order:

     * measure their width (using BitmapText metrics or `measureText`),
     * build lines until we hit max width,
     * assign each token a `{x, y, width, height, sentenceId, t}`.

   Now each token has a precise bounding box we control.

3. Draw:

   * **Default zoom:**

     * Draw text normally.
     * Maybe a faint underline/halo for tokens with strong activations.
   * **Token zoom:**

     * Increase line spacing.
     * For the focused token, draw a cuboid behind it.
     * Under that line, render N concept labels + mini bars/sparkline per token.
   * **Sentence zoom:**

     * Add extra vertical space per sentence.
     * For each sentence:

       * show top-K sentence concepts above the line,
       * draw compressed sparklines across all tokens in that sentence (using the token x positions).
   * **Reply zoom:**

     * Use the full width as time axis,
     * concept tracks are lines that map over all tokens’ x positions,
     * the text can fade/de-emphasize or stay fully visible.

All of this happens in the same coordinate system, so text and tracks never drift apart.

---

## 3. How this lives “inside” Streamlit chat

we can still use `st.chat_message`, but instead of:

```python
with st.chat_message("assistant"):
    st.markdown(reply_text)
    show_visualisation_separately()
```

we do:

```python
with st.chat_message("assistant"):
    timeline_component.timeline_viz(
        text=reply_text,
        data=activation_json,
        key=f"reply-{reply_id}",
    )
```

And our Streamlit component:

* mounts a `<div>` or `<canvas>` that the widget owns,
* passes both the `text` and the `data` into the JS bundle,
* the widget draws *both* the text and the vis.

Visually, it still looks like “an assistant bubble”, it’s just custom-rendered.

---

## 4. How this lives inside OpenWebUI

Same trick:

* In OpenWebUI’s React message renderer, for the assistant, instead of using the default Markdown renderer:

  * pass the message text + activation JSON into our widget wrapper,
  * the widget takes over drawing the entire bubble.

That wrapper can still be a tiny React component:

```tsx
export function HatCatMessage({ text, activations }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (ref.current && window.initHatCatViz) {
      window.initHatCatViz(ref.current, { text, data: activations });
    }
  }, [text, activations]);

  return <div ref={ref} style={{ width: "100%" }} />;
}
```

React is just the host; all the heavy lifting and layout lives in your core widget.

---

## 5. Accessibility with “same-space” constraint

Because we’re drawing text ourself, we *do* lose the natural HTML semantics, but we can recover quite a lot:

* Under the canvas (or absolutely positioned offscreen), we can render:

  ```html
  <div aria-hidden="false" style="position:absolute; left:-9999px;">
    <p>Full reply text here…</p>
    <dl>
      <dt>Top concepts (reply)</dt>
      <dd>helpfulness, safety, empathy, …</dd>
      <dt>Most active sentence:</dt>
      <dd>“…sentence text…”</dd>
    </dl>
  </div>
  ```

* The canvas itself can have `role="img"` + `aria-label="Concept timeline for this reply"`.

* can even add keyboard shortcuts that:

  * move a logical “cursor” over tokens,
  * update an ARIA live region with “Token X: [text], concepts: …”.

So screen-reader users get:

* the exact same text content,
* plus a summary of concept behaviour,
* without needing to interpret the visual timeline.

Given that this is essentially a research/debugger UI, that’s a reasonable level of accessibility.

---

## 6. Summary of the approach

To meet all our constraints:

* ✅ Text and visualisation are in the **same space** and perfectly aligned.
* ✅ Works in **Streamlit chat** and **OpenWebUI**, with one codebase.
* ✅ Doesn’t rely on React for layout; React/Streamlit are just hosts.
* ✅ Gives us GPU acceleration for large activation maps.

I’d do:

1. **Core widget**:

   * TypeScript + PixiJS
   * exports `initHatCatViz(container, { text, data, options })`
   * does text layout + concept timeline rendering.

2. **Streamlit wrapper**:

   * Custom component that passes `text` + `activation_json` to `initHatCatViz`.

3. **OpenWebUI wrapper**:

   * Tiny React component that passes `text` + `activation_json` to `initHatCatViz`.



DATA MODEL 
type ConceptScore = { id: string; score: number };

type TokenInput = {
  id: number;
  text: string;
  sentenceId: number;
  timestep: number;          // 0..N-1
  concepts: ConceptScore[];  // top-k per token
};

type SentenceInput = {
  id: number;
  tokenStart: number;        // index into tokens[]
  tokenEnd: number;          // inclusive
  concepts: ConceptScore[];  // top-k per sentence
};

type ReplyInput = {
  concepts: ConceptScore[];  // top-k per reply
};

type ReplyData = {
  tokens: TokenInput[];
  sentences: SentenceInput[];
  reply: ReplyInput;
};


Coordinate system 

const app = new PIXI.Application({ /* width/height from container */ });

const root = new PIXI.Container();
app.stage.addChild(root);

// Layers
const tokenBgLayer = new PIXI.Container();   // little tiles behind text
const textLayer    = new PIXI.Container();   // token text
const conceptLayer = new PIXI.Container();   // tracks, labels, etc.
const overlayLayer = new PIXI.Container();   // selection outlines, tooltips

root.addChild(tokenBgLayer, conceptLayer, textLayer, overlayLayer);

Everything shares the same x/y coordinate system:

x = horizontal position in pixels.

y = vertical position in pixels, increasing downward.

Zoom/pan will mostly manipulate root.scale and root.position, plus some per-zoom vertical offsets.


Layout Pass 
We want a function that, given the tokens and the available width, returns:

which line each token is on,

and its (x, y, width, height).

type LayoutOptions = {
  paddingX: number;  // left/right
  paddingY: number;  // top
  maxWidth: number;  // container width
  fontSize: number;
  lineHeight: number;  // multiplier, e.g. 1.3
};

const baseStyle = new PIXI.TextStyle({
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: options.fontSize,
});

function measureToken(token: string): number {
  return PIXI.TextMetrics.measureText(token, baseStyle).width;
}

type TokenLayout = {
  token: TokenInput;
  x: number;
  y: number;
  width: number;
  height: number;
  lineIndex: number;
};

type LayoutResult = {
  tokens: TokenLayout[];
  lineCount: number;
  lineHeightPx: number;
};

function layoutTokens(
  reply: ReplyData,
  options: LayoutOptions
): LayoutResult {
  const { paddingX, paddingY, maxWidth, fontSize, lineHeight } = options;

  const lineHeightPx = fontSize * lineHeight;
  const layouts: TokenLayout[] = [];

  let currentLine = 0;
  let cursorX = paddingX;
  let cursorY = paddingY;

  const availableWidth = maxWidth - paddingX * 2;

  for (const tok of reply.tokens) {
    const w = measureToken(tok.text);
    const h = lineHeightPx;

    // + small space between tokens
    const tokenPlusGap = (layouts.length === 0 ? 0 : 4) + w;

    // If this token doesn’t fit on the current line, wrap
    if (cursorX - paddingX + tokenPlusGap > availableWidth) {
      currentLine += 1;
      cursorX = paddingX;
      cursorY = paddingY + currentLine * lineHeightPx;
    }

    // Add a small gap before each token except the first in the line
    if (cursorX > paddingX) cursorX += 4;

    layouts.push({
      token: tok,
      x: cursorX,
      y: cursorY,
      width: w,
      height: h,
      lineIndex: currentLine,
    });

    cursorX += w;
  }

  return {
    tokens: layouts,
    lineCount: currentLine + 1,
    lineHeightPx,
  };
}
This gives  a stable “base layout” in logical coordinates, independent of zoom.

From here on, everything (concept tracks, cuboids, selection highlights) is computed from these token positions.


We’ll often need, for each sentence:

the min/max x across its tokens,

the representative y (e.g. the first line’s y),

and the vertical extent if it spans multiple lines 
type SentenceLayout = {
  sentence: SentenceInput;
  minX: number;
  maxX: number;
  topY: number;
  bottomY: number;
};

function computeSentenceLayouts(
  reply: ReplyData,
  tokenLayouts: TokenLayout[]
): SentenceLayout[] {
  const bySentence = new Map<number, SentenceLayout>();

  for (const tl of tokenLayouts) {
    const sid = tl.token.sentenceId;
    const sMeta = reply.sentences.find((s) => s.id === sid);
    if (!sMeta) continue;

    let acc = bySentence.get(sid);
    if (!acc) {
      acc = {
        sentence: sMeta,
        minX: tl.x,
        maxX: tl.x + tl.width,
        topY: tl.y,
        bottomY: tl.y + tl.height,
      };
      bySentence.set(sid, acc);
    } else {
      acc.minX = Math.min(acc.minX, tl.x);
      acc.maxX = Math.max(acc.maxX, tl.x + tl.width);
      acc.topY = Math.min(acc.topY, tl.y);
      acc.bottomY = Math.max(acc.bottomY, tl.y + tl.height);
    }
  }

  return Array.from(bySentence.values()).sort(
    (a, b) => a.sentence.id - b.sentence.id
  );
}


Let’s define zoom levels from the widget’s POV:
type ZoomLevel = "reply" | "sentence" | "token";
“default chat view” is outside the widget; Streamlit/OpenWebUI handle that.)

5.1 General idea

Base layout: token (x, y) positions from layoutTokens.

Zoom level controls:

vertical spacing multipliers,

which concept data we display,

how much extra vertical “budget” we allocate above/below tokens.

So instead of recomputing line breaks for each zoom, we:

keep the same x-span and line indices,

transform y via a zoom layout function.

5.2 Computing zoomed y positions

You can think of “content blocks”:

each line gets:

base text line (where the tokens live),

optional concept area above or below it.

type ZoomLayoutConfig = {
  textScale: number;      // scale on token text
  lineSpacing: number;    // multiplier over base lineHeightPx
  conceptAreaAbove: number; // px reserved above text line
  conceptAreaBelow: number; // px reserved below text line
};

const ZOOM_CONFIGS: Record<ZoomLevel, ZoomLayoutConfig> = {
  reply: {
    textScale: 1.0,
    lineSpacing: 1.2,
    conceptAreaAbove: 40,   // for per-reply/sentence labels
    conceptAreaBelow: 20,   // small sparklines
  },
  sentence: {
    textScale: 1.05,
    lineSpacing: 1.5,
    conceptAreaAbove: 60,
    conceptAreaBelow: 40,
  },
  token: {
    textScale: 1.15,
    lineSpacing: 2.0,
    conceptAreaAbove: 80,
    conceptAreaBelow: 60,
  },
};


function applyZoomToTokens(
  tokenLayouts: TokenLayout[],
  baseLineHeightPx: number,
  zoom: ZoomLevel
): TokenLayout[] {
  const cfg = ZOOM_CONFIGS[zoom];

  return tokenLayouts.map((tl) => {
    const lineBaseY = tl.lineIndex * baseLineHeightPx * cfg.lineSpacing;
    const zoomedY = lineBaseY; // plus global padding handled elsewhere

    return {
      ...tl,
      y: zoomedY,
      height: baseLineHeightPx * cfg.textScale,
    };
  });
}
x stays the same.

y is recomputed based on line index and zoom-level spacing.

token text sprites are scaled in Y/X by cfg.textScale.

Very high level, ignoring full sprite management:
'''
function renderTokens(layouts: TokenLayout[], cfg: ZoomLayoutConfig) {
  textLayer.removeChildren();
  tokenBgLayer.removeChildren();

  for (const tl of layouts) {
    const bg = new PIXI.Graphics();
    bg.beginFill(0x333333);
    bg.drawRoundedRect(
      tl.x - 2,
      tl.y - tl.height * 0.8,
      tl.width + 4,
      tl.height,
      3
    );
    bg.endFill();
    tokenBgLayer.addChild(bg);

    const text = new PIXI.Text(tl.token.text, baseStyle);
    text.x = tl.x;
    text.y = tl.y - tl.height * 0.8;
    text.scale.set(cfg.textScale);
    // store a reference for hit-testing
    (text as any).tokenId = tl.token.id;

    textLayer.addChild(text);
  }
}
'''
Reply-level concepts (zoom "reply")

Use the token x positions as sample points on the time axis:


'''
function renderReplyConceptTracks(
  reply: ReplyData,
  layouts: TokenLayout[],
  cfg: ZoomLayoutConfig
) {
  conceptLayer.removeChildren();

  const baseY = cfg.conceptAreaAbove; // slightly above first line

  const topConcepts = reply.reply.concepts.slice(0, 5);

  topConcepts.forEach((concept, idx) => {
    const track = new PIXI.Graphics();
    const trackHeight = 10;
    const offsetY = baseY + idx * (trackHeight + 6);

    track.lineStyle(1, 0x00ffff, 0.8);
    layouts.forEach((tl, i) => {
      const scoreForThisToken =
        tl.token.concepts.find((c) => c.id === concept.id)?.score ?? 0;
      const y =
        offsetY - scoreForThisToken * trackHeight; // higher score = higher

      if (i === 0) track.moveTo(tl.x, y);
      else track.lineTo(tl.x, y);
    });

    conceptLayer.addChild(track);

    // Label at left
    const label = new PIXI.Text(concept.id, {
      fontSize: 10,
      fill: 0x88ffff,
    });
    label.x = layouts[0].x;
    label.y = offsetY - trackHeight - 12;
    conceptLayer.addChild(label);
  });
}
'''

Sentence-level concepts (zoom "sentence")

Here we’d use SentenceLayout spans and reply.sentences[..].concepts:

draw a little band above each sentence span (minX..maxX),

render top-K concepts as tiny labels in that band,

optionally draw little per-token bars underneath using the token layouts.

Token-level concepts (zoom "token")

When a token is focused:

find its layout,

reserve a rectangle below that line (using cfg.conceptAreaBelow),

render its top-K concepts with labels and bar lengths proportional to scores.


Hit-testing tokens

Pixi sprites can be interactive:
text.interactive = true;
text.buttonMode = true;
text.on("pointerover", () => selectToken(tl.token.id));
text.on("pointertap", () => zoomToToken(tl.token.id));


selectToken might:

highlight that token’s bg,

show a thin vertical line in concept tracks at that timestep.

zoomToToken might:

switch zoomLevel = "token",

recompute zoomed layout,

animate transition (tween root.scale or y positions).


Zoom control

we can bake a simple state machine:

'''
let zoomLevel: ZoomLevel = "reply";
let focusedSentenceId: number | null = null;
let focusedTokenId: number | null = null;

function setZoom(newZoom: ZoomLevel, opts?: { sentenceId?; tokenId? }) {
  zoomLevel = newZoom;
  focusedSentenceId = opts?.sentenceId ?? null;
  focusedTokenId = opts?.tokenId ?? null;

  // recompute zoomed token layouts
  const zoomedLayouts = applyZoomToTokens(
    baseLayouts.tokens,
    baseLayouts.lineHeightPx,
    zoomLevel
  );

  renderTokens(zoomedLayouts, ZOOM_CONFIGS[zoomLevel]);

  if (zoomLevel === "reply") {
    renderReplyConceptTracks(replyData, zoomedLayouts, ZOOM_CONFIGS.reply);
  } else if (zoomLevel === "sentence") {
    renderSentenceConceptViews(
      replyData,
      zoomedLayouts,
      sentenceLayouts,
      focusedSentenceId
    );
  } else if (zoomLevel === "token") {
    renderTokenConceptView(
      replyData,
      zoomedLayouts,
      focusedTokenId
    );
  }
}
'''

Mouse wheel / pinch zoom can step between levels:

scroll up: reply → sentence → token

scroll down: token → sentence → reply

Or use explicit UI controls.

With this layout approach:

Tokens & text:

are always laid out by your own layoutTokens function,

so you can align any other visual element to them precisely.

Zoom levels:

don’t change word wrapping; just:

change vertical spacing,

decide what concept data to show, and where.

Concept tracks:

are always drawn using the token x positions as the time axis,

so a concept line “passes through” the token that caused it.