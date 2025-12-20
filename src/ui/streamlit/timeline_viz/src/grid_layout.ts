/**
 * Grid-based layout system for timeline visualization.
 *
 * Grid structure:
 * - Each column represents a timestep/token position
 * - Concept tracks flow horizontally and wrap to follow token text
 * - Like a video editor timeline where tracks wrap with the content
 */

import type { ReplyData, ZoomLevel } from './types';

export type GridColumn = {
  x: number;           // X position in pixels
  width: number;       // Width in pixels
  active: boolean;     // true = token/viz, false = label/spacing
  tokenId?: number;    // If active, which token goes here
  sentenceId?: number; // Which sentence this column belongs to (for sentence view)
  paragraphId?: number; // Which paragraph this column belongs to (for paragraph view)
  label?: string;      // If inactive, what label to show
  lineIndex: number;   // Which wrapping line this column is on
};

export type GridRow = {
  y: number;           // Y position in pixels
  height: number;      // Height in pixels
  type: 'token' | 'concept' | 'track-separator';
  lineIndex?: number;  // For multi-line layouts, which line this row belongs to
  trackIndex?: number; // Which concept track (0-4 for top 5)
};

// Maps (tokenId, trackIndex) -> conceptId to show on that track for that token
export type ConceptMapping = Map<string, string>;

export type GridLayout = {
  columns: GridColumn[];
  rows: GridRow[];
  columnWidth: number;
  rowHeight: number;
  wrappedLines: number; // How many lines the tokens wrap across
  conceptMapping: ConceptMapping; // Which concept is active on each track for each token
};

export interface GridLayoutOptions {
  maxWidth: number;
  columnWidth: number;
  rowHeight: number;
  topConceptCount: number;
  paddingX: number;
  paddingY: number;
}

/**
 * Build grid layout based on zoom level.
 * Tokens flow left-to-right and wrap like text. Concept tracks follow the same flow.
 */
export function buildGridLayout(
  reply: ReplyData,
  zoom: ZoomLevel,
  options: GridLayoutOptions
): GridLayout {
  const { maxWidth, columnWidth, rowHeight, topConceptCount, paddingX, paddingY } = options;

  const columns: GridColumn[] = [];
  const rows: GridRow[] = [];

  // Label column width (left margin for concept names)
  const labelColumnWidth = 80;

  // Calculate how many active columns fit in one line
  const availableWidth = maxWidth - labelColumnWidth - paddingX * 2;
  const columnsPerLine = Math.max(1, Math.floor(availableWidth / columnWidth));

  let currentLineIndex = 0;
  let currentColumnInLine = 0;

  if (zoom === 'chat') {
    // Chat view: proportional spacing based on token text length for natural flow
    const baseCharWidth = 8;  // Approximate pixels per character in monospace
    const tokenPadding = 4;   // Small padding between tokens
    let currentX = paddingX + labelColumnWidth;
    let currentLineWidth = 0;

    for (let i = 0; i < reply.tokens.length; i++) {
      const token = reply.tokens[i];
      // Calculate width based on text length (with min/max bounds)
      const textWidth = Math.max(15, Math.min(token.text.length * baseCharWidth, 150));
      const tokenWidth = textWidth + tokenPadding;

      // Check if token fits on current line
      if (currentLineWidth > 0 && currentLineWidth + tokenWidth > availableWidth) {
        // Wrap to next line
        currentLineIndex++;
        currentX = paddingX + labelColumnWidth;
        currentLineWidth = 0;
      }

      columns.push({
        x: currentX,
        width: tokenWidth,
        active: true,
        tokenId: token.id,
        lineIndex: currentLineIndex,
      });

      currentX += tokenWidth;
      currentLineWidth += tokenWidth;
    }
  } else if (zoom === 'reply') {
    // Reply view: tokens flow continuously across lines
    for (let i = 0; i < reply.tokens.length; i++) {
      if (currentColumnInLine >= columnsPerLine) {
        currentLineIndex++;
        currentColumnInLine = 0;
      }

      const x = paddingX + labelColumnWidth + (currentColumnInLine * columnWidth);

      columns.push({
        x,
        width: columnWidth,
        active: true,
        tokenId: reply.tokens[i].id,
        lineIndex: currentLineIndex,
      });

      currentColumnInLine++;
    }
  } else if (zoom === 'paragraph') {
    // Paragraph view: similar to sentence but with gaps at paragraph boundaries
    const labelGapWidth = 120;
    let lastParagraphId: number | undefined = undefined;

    for (let i = 0; i < reply.tokens.length; i++) {
      const token = reply.tokens[i];
      const tokenSentence = reply.sentences.find(s => s.id === token.sentenceId);
      if (!tokenSentence) continue;

      // Find which paragraph this sentence belongs to
      const paragraph = reply.paragraphs.find(
        p => tokenSentence.id >= p.sentenceStart && tokenSentence.id <= p.sentenceEnd
      );
      const paragraphId = paragraph?.id;

      // Insert gap column when paragraph changes
      if (paragraphId !== undefined && paragraphId !== lastParagraphId && lastParagraphId !== undefined) {
        if (currentColumnInLine + 1 > columnsPerLine) {
          currentLineIndex++;
          currentColumnInLine = 0;
        }

        const gapX = paddingX + labelColumnWidth + (currentColumnInLine * columnWidth);
        columns.push({
          x: gapX,
          width: labelGapWidth,
          active: false,
          label: `P${paragraphId + 1}`,
          paragraphId: paragraphId,
          lineIndex: currentLineIndex,
        });

        currentColumnInLine++;
        if (currentColumnInLine >= columnsPerLine) {
          currentLineIndex++;
          currentColumnInLine = 0;
        }
      }

      if (currentColumnInLine >= columnsPerLine) {
        currentLineIndex++;
        currentColumnInLine = 0;
      }

      const x = paddingX + labelColumnWidth + (currentColumnInLine * columnWidth);

      columns.push({
        x,
        width: columnWidth,
        active: true,
        tokenId: token.id,
        sentenceId: token.sentenceId,
        paragraphId: paragraphId,
        lineIndex: currentLineIndex,
      });

      currentColumnInLine++;
      lastParagraphId = paragraphId;
    }
  } else if (zoom === 'sentence') {
    // Sentence view: tokens grouped by sentence with label gaps at boundaries
    const labelGapWidth = 100; // Width for label columns between sentences
    let lastSentenceId: number | undefined = undefined;

    for (let i = 0; i < reply.tokens.length; i++) {
      const token = reply.tokens[i];

      // Check if we're starting a new sentence (and not the first token)
      if (token.sentenceId !== lastSentenceId && lastSentenceId !== undefined) {
        // Insert a label gap column for the new sentence
        // Check if it fits on current line
        if (currentColumnInLine > 0 && currentColumnInLine + 1 <= columnsPerLine) {
          // Fits on current line - add inline label gap
          const x = paddingX + labelColumnWidth + (currentColumnInLine * columnWidth);

          columns.push({
            x,
            width: labelGapWidth,
            active: false,
            label: `S${token.sentenceId}`,
            lineIndex: currentLineIndex,
            sentenceId: token.sentenceId,
          });

          currentColumnInLine++;
        } else {
          // Doesn't fit or we're at the start - wrap to new line
          currentLineIndex++;
          currentColumnInLine = 0;
        }
      }

      // Check if token fits on current line
      if (currentColumnInLine >= columnsPerLine) {
        currentLineIndex++;
        currentColumnInLine = 0;
      }

      const x = paddingX + labelColumnWidth + (currentColumnInLine * columnWidth);

      columns.push({
        x,
        width: columnWidth,
        active: true,
        tokenId: token.id,
        sentenceId: token.sentenceId,
        lineIndex: currentLineIndex,
      });

      currentColumnInLine++;
      lastSentenceId = token.sentenceId;
    }
  } else if (zoom === 'token') {
    // Token view: larger columns for each token to fit concept names
    const tokenColumnWidth = columnWidth * 4; // Increased to 4x for long AI psychology concept names
    const tokensPerLine = Math.max(1, Math.floor(availableWidth / tokenColumnWidth));

    for (let i = 0; i < reply.tokens.length; i++) {
      if (currentColumnInLine >= tokensPerLine) {
        // Wrap to next line
        currentLineIndex++;
        currentColumnInLine = 0;
      }

      const x = paddingX + labelColumnWidth + (currentColumnInLine * tokenColumnWidth);

      columns.push({
        x,
        width: tokenColumnWidth,
        active: true,
        tokenId: reply.tokens[i].id,
        lineIndex: currentLineIndex,
      });

      currentColumnInLine++;
    }
  }

  // Build rows: fixed structure (token row + N concept tracks per wrapped line)
  const numLines = currentLineIndex + 1;
  const conceptMapping = new Map<string, string>();
  let currentY = paddingY;

  // Determine number of concept tracks based on zoom level
  const numConceptTracks = zoom === 'chat' ? 0 : topConceptCount;

  // Build fixed row structure
  for (let line = 0; line < numLines; line++) {
    // Token row for this line
    rows.push({
      y: currentY,
      height: rowHeight,
      type: 'token',
      lineIndex: line,
    });
    currentY += rowHeight;

    // Fixed concept track rows (0 for chat view, topConceptCount for others)
    for (let trackIdx = 0; trackIdx < numConceptTracks; trackIdx++) {
      rows.push({
        y: currentY,
        height: rowHeight * 0.8,
        type: 'concept',
        lineIndex: line,
        trackIndex: trackIdx,
      });
      currentY += rowHeight * 0.8;
    }

    // Add separator between wrapped lines (only if we have concept tracks)
    if (line < numLines - 1 && numConceptTracks > 0) {
      rows.push({
        y: currentY,
        height: rowHeight * 0.3,
        type: 'track-separator',
        lineIndex: line,
      });
      currentY += rowHeight * 0.3;
    }
  }

  // Now populate concept mapping based on zoom level
  if (zoom === 'chat') {
    // Chat view: no concept mapping (clean view)
    // conceptMapping stays empty
  } else if (zoom === 'reply') {
    // Reply view: same concepts for all tokens
    const topConcepts = reply.reply.concepts.slice(0, topConceptCount);

    for (const col of columns) {
      if (!col.active || col.tokenId === undefined) continue;

      for (let trackIdx = 0; trackIdx < topConcepts.length; trackIdx++) {
        const key = `${col.tokenId}-${trackIdx}`;
        conceptMapping.set(key, topConcepts[trackIdx].id);
      }
    }
  } else if (zoom === 'paragraph') {
    // Paragraph view: map each paragraph's concepts to the tracks for its tokens
    for (const paragraph of reply.paragraphs) {
      const paragraphConcepts = paragraph.concepts.slice(0, topConceptCount);
      const paragraphColumns = columns.filter(c => c.paragraphId === paragraph.id);

      for (const col of paragraphColumns) {
        if (!col.active || col.tokenId === undefined) continue;

        for (let trackIdx = 0; trackIdx < paragraphConcepts.length; trackIdx++) {
          const key = `${col.tokenId}-${trackIdx}`;
          conceptMapping.set(key, paragraphConcepts[trackIdx].id);
        }
      }
    }
  } else if (zoom === 'sentence') {
    // Sentence view: map each sentence's concepts to the tracks for its tokens
    for (const sentence of reply.sentences) {
      const sentenceConcepts = sentence.concepts.slice(0, topConceptCount);
      const sentenceColumns = columns.filter(c => c.sentenceId === sentence.id);

      for (const col of sentenceColumns) {
        if (!col.active || col.tokenId === undefined) continue;

        for (let trackIdx = 0; trackIdx < sentenceConcepts.length; trackIdx++) {
          const key = `${col.tokenId}-${trackIdx}`;
          conceptMapping.set(key, sentenceConcepts[trackIdx].id);
        }
      }
    }
  } else if (zoom === 'token') {
    // Token view: map each token's concepts to its own tracks
    for (const token of reply.tokens) {
      const tokenConcepts = token.concepts.slice(0, topConceptCount);

      for (let trackIdx = 0; trackIdx < tokenConcepts.length; trackIdx++) {
        const key = `${token.id}-${trackIdx}`;
        conceptMapping.set(key, tokenConcepts[trackIdx].id);
      }
    }
  }

  return {
    columns,
    rows,
    columnWidth,
    rowHeight,
    wrappedLines: numLines,
    conceptMapping,
  };
}
