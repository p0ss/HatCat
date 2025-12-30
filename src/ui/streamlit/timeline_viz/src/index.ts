/**
 * Main entry point for HatCat timeline visualization widget.
 * Uses grid-based layout system for clean, flexible rendering.
 */

import * as PIXI from 'pixi.js';
import type { ReplyData, VizOptions, ZoomLevel } from './types';
import { buildGridLayout, type GridLayout } from './grid_layout';
import { renderGrid } from './grid_renderer';

export class HatCatTimelineViz {
  private app: PIXI.Application;
  private root: PIXI.Container;

  // Data
  private replyData: ReplyData;
  private maxWidth: number;
  private topConceptCount: number;

  // State
  private currentZoom: ZoomLevel;
  private currentGrid: GridLayout | null = null;

  constructor(
    container: HTMLElement,
    replyData: ReplyData,
    options: VizOptions = {}
  ) {
    this.maxWidth = options.maxWidth ?? container.clientWidth ?? 800;
    this.replyData = replyData;
    this.currentZoom = options.initialZoom ?? 'reply';
    this.topConceptCount = options.topConceptCount ?? 5;

    // Initialize PixiJS
    this.app = new PIXI.Application({
      width: this.maxWidth,
      height: 600,
      backgroundColor: 0x1a1a1a,
      antialias: true,
      resizeTo: container,  // Make canvas resize with container
    });

    container.appendChild(this.app.view as HTMLCanvasElement);

    // Create root container
    this.root = new PIXI.Container();
    this.app.stage.addChild(this.root);

    // Initial render
    this.render();

    // Setup interactions
    this.setupInteractions();

    // Handle window resize
    window.addEventListener('resize', () => {
      this.maxWidth = container.clientWidth;
      this.render();
    });
  }

  private render(): void {
    // Build grid layout for current zoom level
    this.currentGrid = buildGridLayout(this.replyData, this.currentZoom, {
      maxWidth: this.maxWidth,
      columnWidth: 50,  // Width of each token column
      rowHeight: 50,    // Height of each row (token + concepts)
      topConceptCount: this.topConceptCount,
      paddingX: 10,
      paddingY: 10,
    });

    // Render the grid
    renderGrid(this.root, this.replyData, this.currentGrid, this.currentZoom);

    // Update canvas height based on grid
    const totalHeight = this.currentGrid.rows.reduce((sum, row) => sum + row.height, 0) + 100; // +100 for controls
    this.app.renderer.resize(this.app.renderer.width, totalHeight);

    // Notify parent frame of height change (for Streamlit iframe)
    if (window.parent !== window) {
      window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: totalHeight }, '*');
    }
  }

  private setupInteractions(): void {
    // Token hover/click interactions
    this.root.children.forEach((child) => {
      if (child instanceof PIXI.Container) {
        child.children.forEach((grandchild) => {
          if (grandchild instanceof PIXI.Text && (grandchild as any).tokenId !== undefined) {
            const text = grandchild as PIXI.Text;
            const tokenId = (text as any).tokenId;

            text.on('pointerover', () => {
              text.style.fill = 0xffffff;
            });

            text.on('pointerout', () => {
              text.style.fill = 0xf0e6c5;
            });

            text.on('pointertap', () => {
              // Clicking a token in reply/sentence view zooms to token view
              if (this.currentZoom !== 'token') {
                this.setZoom('token');
              }
            });
          }
        });
      }
    });

    // Mouse wheel zoom controls
    const canvas = this.app.view as HTMLCanvasElement;
    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();

      const zoomLevels: ZoomLevel[] = ['chat', 'reply', 'paragraph', 'sentence', 'token'];
      const currentIndex = zoomLevels.indexOf(this.currentZoom);

      if (e.deltaY < 0 && currentIndex < zoomLevels.length - 1) {
        // Zoom in (scroll up)
        this.setZoom(zoomLevels[currentIndex + 1]);
      } else if (e.deltaY > 0 && currentIndex > 0) {
        // Zoom out (scroll down)
        this.setZoom(zoomLevels[currentIndex - 1]);
      }
    });
  }

  public setZoom(zoom: ZoomLevel, options?: { tokenId?: number; sentenceId?: number }): void {
    this.currentZoom = zoom;
    this.render();
    this.setupInteractions();

    // Dispatch custom event for button UI updates
    window.dispatchEvent(new CustomEvent('hatcat:zoomchange', { detail: { zoom } }));
  }

  public destroy(): void {
    this.app.destroy(true, { children: true, texture: true });
  }
}

/**
 * Initialize HatCat timeline visualization in a container.
 * Main entry point for external use (Streamlit, OpenWebUI, etc.)
 */
export function initHatCatViz(
  container: HTMLElement,
  data: { text: string; data: ReplyData; options?: VizOptions }
): HatCatTimelineViz {
  return new HatCatTimelineViz(container, data.data, data.options);
}

// Export types for consumers
export type {
  ReplyData,
  TokenInput,
  SentenceInput,
  ReplyInput,
  ConceptScore,
  ZoomLevel,
  VizOptions,
} from './types';
