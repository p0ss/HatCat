/**
 * Type definitions for HatCat timeline visualization data model.
 * Based on docs/UI_Visualisation_proposal.md
 */

export type ConceptScore = {
  id: string;
  score: number;
  layer?: number;  // Hierarchy level (0-5)
};

export type TokenInput = {
  id: number;
  text: string;
  sentenceId: number;
  timestep: number;
  concepts: ConceptScore[];
};

export type SentenceInput = {
  id: number;
  tokenStart: number;
  tokenEnd: number;
  concepts: ConceptScore[];
};

export type ParagraphInput = {
  id: number;
  sentenceStart: number;
  sentenceEnd: number;
  tokenStart: number;
  tokenEnd: number;
  concepts: ConceptScore[];
};

export type ReplyInput = {
  concepts: ConceptScore[];
};

export type ReplyData = {
  tokens: TokenInput[];
  sentences: SentenceInput[];
  paragraphs: ParagraphInput[];
  reply: ReplyInput;
};

export type ZoomLevel = "chat" | "reply" | "paragraph" | "sentence" | "token";

export type LayoutOptions = {
  paddingX: number;
  paddingY: number;
  maxWidth: number;
  fontSize: number;
  lineHeight: number;
};

export type TokenLayout = {
  token: TokenInput;
  x: number;
  y: number;
  width: number;
  height: number;
  lineIndex: number;
};

export type LayoutResult = {
  tokens: TokenLayout[];
  lineCount: number;
  lineHeightPx: number;
};

export type SentenceLayout = {
  sentence: SentenceInput;
  minX: number;
  maxX: number;
  topY: number;
  bottomY: number;
};

export type ZoomLayoutConfig = {
  textScale: number;
  lineSpacing: number;
  conceptAreaAbove: number;
  conceptAreaBelow: number;
};

export type VizOptions = {
  maxWidth?: number;
  fontSize?: number;
  initialZoom?: ZoomLevel;
  topConceptCount?: number; // Number of concept tracks to show (default: 5)
};
