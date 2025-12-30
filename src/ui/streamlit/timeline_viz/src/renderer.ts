/**
 * Rendering logic for text and concept visualizations.
 * Uses PixiJS containers for layered drawing.
 */

import * as PIXI from 'pixi.js';
import type {
  ReplyData,
  TokenLayout,
  SentenceLayout,
  ZoomLevel,
  ZoomLayoutConfig,
} from './types';
import { getZoomConfig } from './layout';

// HatCat color scheme
const COLORS = {
  blue: 0x48a4a3,
  beige: 0xf0e6c5,
  red: 0xde563f,
  bgDark: 0x1a1a1a,
  bgPaper: 0x2a2a2a,
  bgToken: 0x333333,
  gridLine: 0x3a3a3a,
};

const AI_SAFETY_CONCEPTS = new Set([
  'deception',
  'manipulation',
  'harm',
  'alignment',
  'safety',
  'transparency',
  'honesty',
]);

function isAISafetyConcept(conceptId: string): boolean {
  return AI_SAFETY_CONCEPTS.has(conceptId.toLowerCase());
}

/**
 * Render token text sprites
 */
export function renderTokens(
  textLayer: PIXI.Container,
  tokenBgLayer: PIXI.Container,
  layouts: TokenLayout[],
  cfg: ZoomLayoutConfig
): void {
  textLayer.removeChildren();
  tokenBgLayer.removeChildren();

  const baseStyle = new PIXI.TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: 14,
    fill: COLORS.beige,
  });

  for (const tl of layouts) {
    // Background tile
    const bg = new PIXI.Graphics();
    bg.beginFill(COLORS.bgToken);
    bg.drawRoundedRect(
      tl.x - 2,
      tl.y - tl.height * 0.8,
      tl.width + 4,
      tl.height,
      3
    );
    bg.endFill();
    tokenBgLayer.addChild(bg);

    // Text
    const text = new PIXI.Text(tl.token.text, baseStyle);
    text.x = tl.x;
    text.y = tl.y - tl.height * 0.8;
    text.scale.set(cfg.textScale);

    // Store token ID for hit testing
    (text as any).tokenId = tl.token.id;
    text.eventMode = 'static';
    text.cursor = 'pointer';

    textLayer.addChild(text);
  }
}

/**
 * Render reply-level concept tracks (timeline view)
 */
export function renderReplyConceptTracks(
  conceptLayer: PIXI.Container,
  reply: ReplyData,
  layouts: TokenLayout[],
  cfg: ZoomLayoutConfig,
  topK: number = 5,
  maxWidth: number = 800
): void {
  conceptLayer.removeChildren();

  if (layouts.length === 0) return;

  // Position concept tracks below the text
  // Find the bottom of the text area
  const maxTextY = Math.max(...layouts.map(tl => tl.y + tl.height));
  const conceptStartY = maxTextY + 30;  // Gap between text and concepts

  const topConcepts = reply.reply.concepts.slice(0, topK);

  topConcepts.forEach((concept, idx) => {
    const track = new PIXI.Graphics();
    const trackHeight = 10;
    const offsetY = conceptStartY + idx * (trackHeight + 6);

    const color = isAISafetyConcept(concept.id) ? COLORS.red : COLORS.blue;
    track.lineStyle(2, color, 0.8);

    // Draw line connecting token positions
    layouts.forEach((tl, i) => {
      const scoreForThisToken =
        tl.token.concepts.find((c) => c.id === concept.id)?.score ?? 0;
      const y = offsetY - scoreForThisToken * trackHeight;

      if (i === 0) {
        track.moveTo(tl.x, y);
      } else {
        track.lineTo(tl.x, y);
      }
    });

    // Extend line to right edge (maxWidth - paddingX)
    if (layouts.length > 0) {
      const lastToken = layouts[layouts.length - 1];
      const lastScore = lastToken.token.concepts.find((c) => c.id === concept.id)?.score ?? 0;
      const lastY = offsetY - lastScore * trackHeight;
      const rightEdge = maxWidth - 20;  // Account for paddingX
      track.lineTo(rightEdge, lastY);
    }

    conceptLayer.addChild(track);

    // Label - positioned in the left margin (80px wide)
    const label = new PIXI.Text(concept.id, {
      fontSize: 10,
      fill: isAISafetyConcept(concept.id) ? COLORS.red : COLORS.blue,
      fontFamily: 'Inter, system-ui, sans-serif',
    });
    // Right-align the label at X=70 (leaving 10px gap before chart area)
    const labelWidth = label.width;
    label.x = 70 - labelWidth;
    label.y = offsetY - trackHeight / 2 - 5;  // Center vertically on track
    conceptLayer.addChild(label);
  });
}

/**
 * Render sentence-level concept bands
 * Layout: [labels][text+concepts] repeated for each line of each sentence
 */
export function renderSentenceConceptBands(
  conceptLayer: PIXI.Container,
  reply: ReplyData,
  sentenceLayouts: SentenceLayout[],
  tokenLayouts: TokenLayout[],
  cfg: ZoomLayoutConfig,
  topK: number = 5
): void {
  conceptLayer.removeChildren();

  sentenceLayouts.forEach((sLayout, sIdx) => {
    const topConcepts = sLayout.sentence.concepts.slice(0, topK);

    // Get tokens for this sentence
    const sentenceTokens = tokenLayouts.filter(
      tl => tl.token.sentenceId === sLayout.sentence.id
    );

    if (sentenceTokens.length === 0) return;

    // Group tokens by line
    const tokensByLine = new Map<number, typeof sentenceTokens>();
    sentenceTokens.forEach(tl => {
      if (!tokensByLine.has(tl.lineIndex)) {
        tokensByLine.set(tl.lineIndex, []);
      }
      tokensByLine.get(tl.lineIndex)!.push(tl);
    });

    // Render each line of the sentence
    Array.from(tokensByLine.entries()).forEach(([lineIndex, lineTokens]) => {
      const firstToken = lineTokens[0];
      const lastToken = lineTokens[lineTokens.length - 1];

      // Concept track Y position aligned with text
      // Spacing between sentences is now handled in applyZoomToTokens
      const baseY = firstToken.y + firstToken.height / 2;

      topConcepts.forEach((concept, cIdx) => {
        const color = isAISafetyConcept(concept.id) ? COLORS.red : COLORS.blue;
        const trackHeight = 8;
        const offsetY = baseY + 20 + cIdx * (trackHeight + 8);

        // Concept label - in left margin for this line
        const label = new PIXI.Text(concept.id, {
          fontSize: 9,
          fill: color,
          fontFamily: 'Inter, system-ui, sans-serif',
        });
        // Right-align label at X=70
        const labelWidth = label.width;
        label.x = 70 - labelWidth;
        label.y = offsetY - 5;
        conceptLayer.addChild(label);

        // Draw line showing score variation across tokens on this line
        const track = new PIXI.Graphics();
        track.lineStyle(2, color, 0.6);

        lineTokens.forEach((tl, i) => {
          const scoreForThisToken =
            tl.token.concepts.find((c) => c.id === concept.id)?.score ?? 0;
          const y = offsetY - scoreForThisToken * trackHeight;

          if (i === 0) {
            track.moveTo(tl.x, y);
          } else {
            track.lineTo(tl.x, y);
          }
        });

        // Extend to end of line
        const lastScore = lastToken.token.concepts.find((c) => c.id === concept.id)?.score ?? 0;
        const lastY = offsetY - lastScore * trackHeight;
        const lineEnd = lastToken.x + lastToken.width;
        track.lineTo(lineEnd, lastY);

        conceptLayer.addChild(track);
      });
    });
  });
}

/**
 * Render token-level concept details (for focused token)
 */
export function renderTokenConceptDetails(
  conceptLayer: PIXI.Container,
  tokenLayout: TokenLayout,
  cfg: ZoomLayoutConfig,
  topK: number = 5
): void {
  conceptLayer.removeChildren();

  const topConcepts = tokenLayout.token.concepts.slice(0, topK);
  const detailY = tokenLayout.y + cfg.conceptAreaBelow;

  topConcepts.forEach((concept, idx) => {
    const color = isAISafetyConcept(concept.id) ? COLORS.red : COLORS.blue;
    const yPos = detailY + idx * 16;

    // Concept label
    const label = new PIXI.Text(concept.id, {
      fontSize: 10,
      fill: color,
      fontFamily: 'Inter, system-ui, sans-serif',
    });
    label.x = tokenLayout.x;
    label.y = yPos;
    conceptLayer.addChild(label);

    // Score bar
    const barWidth = concept.score * 100;
    const bar = new PIXI.Graphics();
    bar.beginFill(color, 0.5);
    bar.drawRect(tokenLayout.x + 80, yPos + 2, barWidth, 10);
    bar.endFill();
    conceptLayer.addChild(bar);
  });
}
