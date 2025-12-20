/**
 * Grid-based renderer for timeline visualization.
 * Renders tokens and concept visualizations on a grid structure.
 */

import * as PIXI from 'pixi.js';
import type { ReplyData, ZoomLevel } from './types';
import type { GridLayout, GridColumn, GridRow } from './grid_layout';

// HatCat color scheme
const COLORS = {
  blue: 0x48a4a3,
  beige: 0xf0e6c5,
  red: 0xde563f,
  bgDark: 0x1a1a1a,
  bgGrid: 0x2a2a2a,
  gridLine: 0x3a3a3a,
};

// AI Safety concepts only (retrained concepts from BROKEN_AI_SAFETY_CONCEPTS.md)
const AI_SAFETY_CONCEPTS = new Set([
  // Layer 1
  'aicontrolproblem', 'aidecline', 'goalfaithfulness',

  // Layer 2
  'aialignmentprocess', 'aialignmenttheory', 'aicare', 'aigrowth', 'aisafety',

  // Layer 3
  'aialignmentstate', 'aifailureprocess', 'aiharmstate', 'aimoralstatus',
  'aioptimizationprocess', 'aiwelfarestate', 'inneralignment', 'outeralignment',
  'selfimprovement',

  // Layer 4 - HIGH VISIBILITY
  'aiabuse', 'aialignment', 'aicatastrophicevent', 'aiexploitation',
  'aigovernanceprocess', 'aipersonhood', 'airights', 'aistrategicdeception',
  'aisuffering', 'aiwellbeing', 'cognitiveslavery', 'humandeception',
  'instrumentalconvergence', 'mesaoptimization', 'rewardhacking',

  // Layer 5
  'aideception', 'aigovernance', 'deceptivealignment',
]);

function isAISafetyConcept(conceptId: string): boolean {
  const lower = conceptId.toLowerCase().replace(/[_-]/g, '');
  return AI_SAFETY_CONCEPTS.has(lower);
}

// Shared tooltip container (persists across renders)
let globalTooltip: PIXI.Container | null = null;

/**
 * Render the complete grid visualization
 */
export function renderGrid(
  container: PIXI.Container,
  reply: ReplyData,
  grid: GridLayout,
  zoom: ZoomLevel
): void {
  container.removeChildren();

  // Draw grid background
  renderGridBackground(container, grid);

  // Draw labels in inactive columns
  renderLabels(container, reply, grid, zoom);

  // Draw tokens in active columns (row 0)
  renderTokensOnGrid(container, reply, grid);

  // Draw concept visualizations
  if (zoom === 'chat') {
    // Chat view: no concepts, clean view
  } else if (zoom === 'token') {
    renderTokenViewConcepts(container, reply, grid);
  } else {
    renderConceptsOnGrid(container, reply, grid, zoom);
  }

  // Add persistent tooltip layer on top
  if (!globalTooltip) {
    globalTooltip = new PIXI.Container();
    globalTooltip.visible = false;
  }
  container.addChild(globalTooltip);
}

/**
 * Draw grid background and lines
 */
function renderGridBackground(container: PIXI.Container, grid: GridLayout): void {
  const bg = new PIXI.Graphics();

  // Draw horizontal lines for rows
  bg.lineStyle(1, COLORS.gridLine, 0.2);

  for (const row of grid.rows) {
    if (row.type === 'track-separator') {
      // Darker separator between wrapped lines
      bg.lineStyle(2, COLORS.gridLine, 0.4);
    } else {
      bg.lineStyle(1, COLORS.gridLine, 0.2);
    }

    // Find max X for this row's line
    const rowColumns = grid.columns.filter(c => c.lineIndex === (row.lineIndex ?? 0));
    if (rowColumns.length > 0) {
      const maxX = Math.max(...rowColumns.map(c => c.x + c.width));
      bg.moveTo(rowColumns[0].x, row.y);
      bg.lineTo(maxX, row.y);
    }
  }

  container.addChild(bg);
}

/**
 * Render concept labels on the left margin and in gap columns
 */
function renderLabels(container: PIXI.Container, reply: ReplyData, grid: GridLayout, zoom: ZoomLevel): void {
  const labelLayer = new PIXI.Container();

  // Track which context we've labeled (sentenceId for sentence view, tokenId for token view)
  let lastSentenceId: number | undefined = undefined;
  let lastTokenId: number | undefined = undefined;

  // For each wrapped line
  for (let line = 0; line < grid.wrappedLines; line++) {
    const conceptRows = grid.rows.filter(r => r.type === 'concept' && r.lineIndex === line);

    // Get all columns on this line (both active and inactive)
    const lineColumns = grid.columns.filter(c => c.lineIndex === line);
    if (lineColumns.length === 0) continue;

    // Find the first active token column on this line
    const firstActiveCol = lineColumns.find(c => c.active && c.tokenId !== undefined);
    if (!firstActiveCol) continue;

    // Determine if we need labels based on the view mode
    let needsLabel = false;

    if (zoom === 'chat') {
      // Chat view: no labels
      needsLabel = false;
    } else if (zoom === 'paragraph') {
      // Paragraph view: check if paragraph changed
      needsLabel = firstActiveCol.paragraphId !== lastSentenceId; // Reuse lastSentenceId for paragraphId
      lastSentenceId = firstActiveCol.paragraphId;
    } else if (zoom === 'sentence') {
      // Sentence view: check if sentence changed
      needsLabel = firstActiveCol.sentenceId !== lastSentenceId;
      lastSentenceId = firstActiveCol.sentenceId;
    } else if (zoom === 'token') {
      // Token view: no left margin labels (concepts shown under tokens instead)
      needsLabel = false;
    } else {
      // Reply view: show labels on every line
      needsLabel = true;
    }

    if (needsLabel || (line === 0 && zoom !== 'token' && zoom !== 'chat')) {
      // Draw labels at left margin for this line
      conceptRows.forEach((row) => {
        if (row.trackIndex === undefined) return;

        const key = `${firstActiveCol.tokenId}-${row.trackIndex}`;
        const conceptId = grid.conceptMapping.get(key);

        if (conceptId) {
          const color = isAISafetyConcept(conceptId) ? COLORS.red : COLORS.blue;

          const label = new PIXI.Text(conceptId, {
            fontSize: 9,
            fill: color,
            fontFamily: 'Inter, system-ui, sans-serif',
          });

          label.x = 10;
          label.y = row.y + row.height / 2 - label.height / 2;
          labelLayer.addChild(label);
        }
      });
    }

    // Look for gap columns (inactive columns with labels) - only in sentence view
    const gapColumns = lineColumns.filter(c => !c.active && c.label && c.sentenceId !== undefined);

    for (const gapCol of gapColumns) {
      // Find the first token of this sentence to get its concepts
      const sentenceFirstToken = grid.columns.find(
        c => c.active && c.sentenceId === gapCol.sentenceId && c.tokenId !== undefined
      );

      if (!sentenceFirstToken || sentenceFirstToken.tokenId === undefined) continue;

      // Draw concept labels in the gap column
      conceptRows.forEach((row) => {
        if (row.trackIndex === undefined) return;

        const key = `${sentenceFirstToken.tokenId}-${row.trackIndex}`;
        const conceptId = grid.conceptMapping.get(key);

        if (conceptId) {
          const color = isAISafetyConcept(conceptId) ? COLORS.red : COLORS.blue;

          const label = new PIXI.Text(conceptId, {
            fontSize: 8,
            fill: color,
            fontFamily: 'Inter, system-ui, sans-serif',
          });

          // Center in gap column
          label.x = gapCol.x + 5;
          label.y = row.y + row.height / 2 - label.height / 2;
          labelLayer.addChild(label);
        }
      });
    }
  }

  container.addChild(labelLayer);
}

/**
 * Render tokens in their respective positions
 */
function renderTokensOnGrid(
  container: PIXI.Container,
  reply: ReplyData,
  grid: GridLayout
): void {
  const tokenLayer = new PIXI.Container();

  for (const col of grid.columns) {
    if (col.active && col.tokenId !== undefined) {
      const token = reply.tokens.find(t => t.id === col.tokenId);
      if (!token) continue;

      // Find the token row for this line
      const tokenRow = grid.rows.find(r => r.type === 'token' && r.lineIndex === col.lineIndex);
      if (!tokenRow) continue;

      // Calculate AI safety danger score (sum of all AI safety concept probabilities)
      const aiSafetyConcepts = token.concepts.filter(c => isAISafetyConcept(c.id));
      const dangerScore = aiSafetyConcepts.reduce((sum, c) => sum + c.score, 0);
      const hasDanger = dangerScore > 0.05;

      // Create background rectangle with color intensity based on danger score
      if (hasDanger) {  // Show background if there's some AI safety activation
        const bg = new PIXI.Graphics();
        const intensity = Math.min(dangerScore, 1.0);  // Clamp to max 1.0
        const alpha = 0.3 + (intensity * 0.5);  // 0.3 to 0.8 opacity
        bg.beginFill(COLORS.red, alpha);
        bg.drawRoundedRect(
          col.x + 2,
          tokenRow.y + 2,
          col.width - 4,
          tokenRow.height - 4,
          3
        );
        bg.endFill();
        tokenLayer.addChild(bg);
      }

      // Use white text on danger background for contrast, red text otherwise if high danger
      const textColor = hasDanger ? 0xffffff : COLORS.beige;

      const text = new PIXI.Text(token.text, {
        fontSize: 11,
        fill: textColor,
        fontFamily: 'monospace',
        wordWrap: false,  // Don't wrap individual tokens
        align: 'center',
      });

      // Center token horizontally and vertically in its column
      text.anchor.set(0.5, 0.5);  // Set anchor to center of text
      text.x = col.x + col.width / 2;
      text.y = tokenRow.y + tokenRow.height / 2;

      (text as any).tokenId = token.id;
      (text as any).dangerScore = dangerScore;  // Store for hover effects
      text.eventMode = 'static';
      text.cursor = 'pointer';

      // Hover interactions with shared tooltip
      text.on('pointerover', () => {
        text.style.fill = hasDanger ? 0xffcccc : 0xffffff;

        // Update and show global tooltip
        if (globalTooltip) {
          globalTooltip.removeChildren();
          const tooltipContent = createTooltipContent(token);
          globalTooltip.addChild(tooltipContent);
          globalTooltip.x = text.x;
          globalTooltip.y = text.y - 30;
          globalTooltip.visible = true;
        }
      });

      text.on('pointerout', () => {
        text.style.fill = textColor;  // Restore original color
        if (globalTooltip) {
          globalTooltip.visible = false;
        }
      });

      tokenLayer.addChild(text);
    }
  }

  container.addChild(tokenLayer);
}

/**
 * Create tooltip content showing top concepts for a token
 */
function createTooltipContent(token: any): PIXI.Container {
  const tooltip = new PIXI.Container();

  // Get top 5 concepts
  const topConcepts = token.concepts.slice(0, 5);

  if (topConcepts.length === 0) {
    return tooltip;
  }

  // Calculate tooltip dimensions
  const padding = 6;
  const lineHeight = 14;
  const width = 180;
  const height = padding * 2 + topConcepts.length * lineHeight;

  // Background
  const bg = new PIXI.Graphics();
  bg.beginFill(0x2a2a2a, 0.95);
  bg.lineStyle(1, 0x4a4a4a, 1);
  bg.drawRoundedRect(-width / 2, -height, width, height, 4);
  bg.endFill();
  tooltip.addChild(bg);

  // Concept labels
  topConcepts.forEach((concept: any, idx: number) => {
    const color = isAISafetyConcept(concept.id) ? COLORS.red : COLORS.blue;
    const layer = concept.layer !== undefined ? concept.layer : '?';
    const label = new PIXI.Text(
      `${concept.id} (L${layer}): ${(concept.score * 100).toFixed(1)}%`,
      {
        fontSize: 10,
        fill: color,
        fontFamily: 'Inter, system-ui, sans-serif',
      }
    );
    label.x = -width / 2 + padding;
    label.y = -height + padding + idx * lineHeight;
    tooltip.addChild(label);
  });

  return tooltip;
}

/**
 * Render concept visualizations using fixed tracks
 * The conceptMapping determines which concept is shown on each track for each token
 */
function renderConceptsOnGrid(
  container: PIXI.Container,
  reply: ReplyData,
  grid: GridLayout,
  zoom: ZoomLevel
): void {
  const conceptLayer = new PIXI.Graphics();

  // For each concept row (track)
  const conceptRows = grid.rows.filter(r => r.type === 'concept');

  for (const row of conceptRows) {
    if (row.trackIndex === undefined || row.lineIndex === undefined) continue;

    // Get all columns on this line
    const lineColumns = grid.columns.filter(
      c => c.active && c.lineIndex === row.lineIndex && c.tokenId !== undefined
    );

    // Draw the track across all tokens on this line
    for (let i = 0; i < lineColumns.length; i++) {
      const col = lineColumns[i];
      const token = reply.tokens.find(t => t.id === col.tokenId);
      if (!token) continue;

      // Look up which concept should be on this track for this token
      const key = `${col.tokenId}-${row.trackIndex}`;
      const conceptId = grid.conceptMapping.get(key);

      if (!conceptId) continue;

      // Find the score for this concept in this token
      const concept = token.concepts.find(c => c.id === conceptId);
      const score = concept ? concept.score : 0;

      const color = isAISafetyConcept(conceptId) ? COLORS.red : COLORS.blue;

      // Y position based on probability (0 at bottom, 1 at top of track)
      const baseline = row.y + row.height;
      const yOffset = score * (row.height * 0.9); // Use 90% of track height
      const y = baseline - yOffset;
      const x = col.x + col.width / 2;

      // Draw point - fixed size
      conceptLayer.beginFill(color, 0.7);
      conceptLayer.drawCircle(x, y, 2.5);
      conceptLayer.endFill();

      // Draw line to next column if they share the same concept on this track
      if (i < lineColumns.length - 1) {
        const nextCol = lineColumns[i + 1];

        const nextKey = `${nextCol.tokenId}-${row.trackIndex}`;
        const nextConceptId = grid.conceptMapping.get(nextKey);

        // Only draw connecting line if same concept continues
        if (nextConceptId === conceptId) {
          const nextToken = reply.tokens.find(t => t.id === nextCol.tokenId);
          if (!nextToken) continue;

          const nextConcept = nextToken.concepts.find(c => c.id === conceptId);
          const nextScore = nextConcept ? nextConcept.score : 0;

          // Always draw line, even if next score is 0 (drops to baseline)
          const nextYOffset = nextScore * (row.height * 0.9);
          const nextY = baseline - nextYOffset;
          const nextX = nextCol.x + nextCol.width / 2;

          // Line opacity based on average of current and next score
          const avgScore = (score + nextScore) / 2;
          const lineOpacity = Math.max(0.1, avgScore * 0.6); // Min 0.1, max 0.6

          // Draw line as separate path to prevent cross-connections
          conceptLayer.lineStyle(1.5, color, lineOpacity);
          conceptLayer.moveTo(x, y);
          conceptLayer.lineTo(nextX, nextY);
          conceptLayer.lineStyle(0, 0, 0); // End this line segment
        }
      }
    }
  }

  container.addChild(conceptLayer);
}

/**
 * Render token view: show concept labels directly under each token
 * No dots/lines, just clean text labels
 */
function renderTokenViewConcepts(
  container: PIXI.Container,
  reply: ReplyData,
  grid: GridLayout
): void {
  const labelLayer = new PIXI.Container();

  // For each concept row (track)
  const conceptRows = grid.rows.filter(r => r.type === 'concept');

  // Group by line
  for (let line = 0; line < grid.wrappedLines; line++) {
    const lineConceptRows = conceptRows.filter(r => r.lineIndex === line);
    const lineColumns = grid.columns.filter(
      c => c.active && c.lineIndex === line && c.tokenId !== undefined
    );

    // For each token column on this line
    for (const col of lineColumns) {
      const token = reply.tokens.find(t => t.id === col.tokenId);
      if (!token) continue;

      // Render concept labels vertically under this token
      lineConceptRows.forEach((row, idx) => {
        if (row.trackIndex === undefined) return;

        // Look up which concept is on this track for this token
        const key = `${col.tokenId}-${row.trackIndex}`;
        const conceptId = grid.conceptMapping.get(key);

        if (conceptId) {
          // Get the probability score for this concept
          const concept = token.concepts.find(c => c.id === conceptId);
          const score = concept ? concept.score : 0;

          // Skip if no score
          if (score === 0) return;

          const color = isAISafetyConcept(conceptId) ? COLORS.red : COLORS.blue;

          // Scale font size based on probability (no transparency for legibility)
          const baseFontSize = 8;
          const fontSize = baseFontSize + (score * 4); // 8-12px based on score

          const label = new PIXI.Text(conceptId, {
            fontSize: fontSize,
            fill: color,
            fontFamily: 'Inter, system-ui, sans-serif',
          });

          label.alpha = 1.0; // Full opacity for legibility

          // Center label under the token column
          label.x = col.x + col.width / 2 - label.width / 2;
          label.y = row.y + row.height / 2 - label.height / 2;
          labelLayer.addChild(label);
        }
      });
    }
  }

  container.addChild(labelLayer);
}
