/**
 * Layout engine for token positioning and sentence bounds.
 * Implements the layout system from docs/UI_Visualisation_proposal.md
 */

import * as PIXI from 'pixi.js';
import type {
  ReplyData,
  LayoutOptions,
  TokenLayout,
  LayoutResult,
  SentenceLayout,
  ZoomLevel,
  ZoomLayoutConfig,
} from './types';

const ZOOM_CONFIGS: Record<ZoomLevel, ZoomLayoutConfig> = {
  chat: {
    textScale: 1.0,
    lineSpacing: 1.2,
    conceptAreaAbove: 0,
    conceptAreaBelow: 0,
  },
  reply: {
    textScale: 1.0,
    lineSpacing: 1.2,
    conceptAreaAbove: 40,
    conceptAreaBelow: 20,
  },
  paragraph: {
    textScale: 1.0,
    lineSpacing: 1.3,
    conceptAreaAbove: 50,
    conceptAreaBelow: 30,
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

export function getZoomConfig(zoom: ZoomLevel): ZoomLayoutConfig {
  return ZOOM_CONFIGS[zoom];
}

/**
 * Measure token width using PIXI TextMetrics
 */
function measureToken(token: string, style: PIXI.TextStyle): number {
  return PIXI.TextMetrics.measureText(token, style).width;
}

/**
 * Layout tokens into lines with word wrapping.
 * Returns base layout before zoom transformations.
 */
export function layoutTokens(
  reply: ReplyData,
  options: LayoutOptions
): LayoutResult {
  const { paddingX, paddingY, maxWidth, fontSize, lineHeight } = options;

  const baseStyle = new PIXI.TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize,
  });

  const lineHeightPx = fontSize * lineHeight;
  const layouts: TokenLayout[] = [];

  // Reserve 80px on the left for concept labels
  const labelWidth = 80;
  let currentLine = 0;
  let cursorX = labelWidth;  // Start after label area
  let cursorY = paddingY;

  const availableWidth = maxWidth - labelWidth - paddingX;

  for (let i = 0; i < reply.tokens.length; i++) {
    const tok = reply.tokens[i];
    const w = measureToken(tok.text, baseStyle);
    const h = lineHeightPx;

    // Check if we need to wrap (only check width, gap will be added after)
    const isFirstOnLine = (cursorX === labelWidth);
    const gap = isFirstOnLine ? 0 : 4;

    // Prevent single-character punctuation from wrapping alone
    const isPunctuation = tok.text.trim().length === 1 && /[.!?,;:]/.test(tok.text.trim());
    const shouldWrap = cursorX + gap + w > labelWidth + availableWidth;

    if (shouldWrap && !isPunctuation) {
      currentLine += 1;
      cursorX = labelWidth;
      cursorY = paddingY + currentLine * lineHeightPx;
    }

    // Add gap before token (except first in line)
    const isNowFirstOnLine = (cursorX === labelWidth);
    if (!isNowFirstOnLine) {
      cursorX += 4;
    }

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

/**
 * Compute sentence bounding boxes from token layouts
 */
export function computeSentenceLayouts(
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

/**
 * Apply zoom transformation to token layouts.
 * Adjusts Y positions and heights based on zoom level.
 * For sentence mode, adds extra spacing between sentences.
 */
export function applyZoomToTokens(
  tokenLayouts: TokenLayout[],
  baseLineHeightPx: number,
  zoom: ZoomLevel
): TokenLayout[] {
  const cfg = ZOOM_CONFIGS[zoom];

  // Text comes first (top), then concept area below
  // Add some top padding for text
  const paddingY = 20;

  // For sentence mode, add spacing between sentences
  const sentenceGap = zoom === 'sentence' ? 60 : 0;

  // Track sentence changes per line for adding gaps
  const lineToSentenceGap = new Map<number, number>();

  if (sentenceGap > 0) {
    // Sort tokens by line index to process in order
    const sortedTokens = [...tokenLayouts].sort((a, b) => {
      if (a.lineIndex !== b.lineIndex) return a.lineIndex - b.lineIndex;
      return a.token.id - b.token.id;
    });

    let prevLineIndex = -1;
    let prevSentenceId = -1;
    let gapsSoFar = 0;

    for (const tl of sortedTokens) {
      // If we're on a new line and the sentence ID changed from previous line
      if (tl.lineIndex !== prevLineIndex) {
        if (prevLineIndex >= 0 && tl.token.sentenceId !== prevSentenceId) {
          gapsSoFar++;
        }
        lineToSentenceGap.set(tl.lineIndex, gapsSoFar);
        prevLineIndex = tl.lineIndex;
        prevSentenceId = tl.token.sentenceId;
      }
    }
  }

  return tokenLayouts.map((tl) => {
    const gapCount = lineToSentenceGap.get(tl.lineIndex) || 0;
    const lineBaseY = paddingY +
                     tl.lineIndex * baseLineHeightPx * cfg.lineSpacing +
                     gapCount * sentenceGap;

    return {
      ...tl,
      y: lineBaseY,
      height: baseLineHeightPx * cfg.textScale,
    };
  });
}
