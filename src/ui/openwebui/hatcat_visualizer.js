/**
 * HatCat Divergence Visualizer - OpenWebUI Custom Function
 *
 * Provides real-time token-level divergence visualization with:
 * - Sunburst color coding based on concept ontology
 * - Hover tooltips showing divergence details
 * - Click-to-steer functionality for concept amplification/suppression
 * - Steering panel for managing active steerings
 */

// ============================================================================
// Configuration
// ============================================================================

const HATCAT_API_BASE = 'http://localhost:8765/v1';
const DEFAULT_SESSION_ID = 'default';

// ============================================================================
// Color Utilities
// ============================================================================

/**
 * Calculate relative luminance of a color (WCAG formula)
 * @param {string} hexColor - Hex color code (e.g., "#FF5733")
 * @returns {number} Luminance value (0-1)
 */
function getLuminance(hexColor) {
  const hex = hexColor.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16) / 255;
  const g = parseInt(hex.substr(2, 2), 16) / 255;
  const b = parseInt(hex.substr(4, 2), 16) / 255;

  const srgb = [r, g, b].map(val => {
    return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
  });

  return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
}

/**
 * Get contrasting text color for a background
 * @param {string} bgColor - Background hex color
 * @returns {string} Text color (#000000 or #ffffff)
 */
function getContrastingTextColor(bgColor) {
  const luminance = getLuminance(bgColor);
  return luminance > 0.5 ? '#000000' : '#ffffff';
}

// ============================================================================
// Token Rendering
// ============================================================================

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Render a token with divergence coloring and metadata
 * @param {string} token - Token text
 * @param {object} metadata - Token metadata from API
 * @returns {string} HTML string for colored token
 */
function renderToken(token, metadata) {
  if (!metadata || !metadata.divergence) {
    // No metadata - render plain token
    return escapeHtml(token);
  }

  const color = metadata.color || '#808080';
  const textColor = getContrastingTextColor(color);
  const divergence = metadata.divergence;
  const steering = metadata.steering || { active: false, steerings: [] };

  // Create tooltip content
  const tooltipData = {
    token: token,
    max_divergence: divergence.max_divergence || 0,
    top_divergences: divergence.top_divergences || [],
    activation_detections: divergence.activation_detections || [],
    text_detections: divergence.text_detections || [],
    palette: metadata.palette || [],
    steering: steering
  };

  // Add steering indicator if active
  const steeringIndicator = steering.active ?
    '<span class="steering-indicator">⚡</span>' : '';

  return `<span
    class="hatcat-token"
    style="
      background-color: ${color};
      color: ${textColor};
      padding: 2px 4px;
      margin: 0 1px;
      border-radius: 3px;
      cursor: help;
      position: relative;
      display: inline-block;
      transition: transform 0.1s;
    "
    data-tooltip='${JSON.stringify(tooltipData)}'
    onmouseover="HatCat.showTooltip(this, event)"
    onmouseout="HatCat.hideTooltip()"
    oncontextmenu="HatCat.showSteeringMenu(this, event); return false;">
    ${steeringIndicator}${escapeHtml(token)}
  </span>`;
}

// ============================================================================
// HatCat Main Object
// ============================================================================

window.HatCat = {
  sessionId: DEFAULT_SESSION_ID,
  activeTooltip: null,
  steeringMenu: null,
  steerings: [],

  /**
   * Initialize HatCat visualizer
   */
  init() {
    console.log('🎩 HatCat Visualizer initialized');
    this.createTooltipElement();
    this.createSteeringMenu();
    this.createSteeringPanel();
    this.loadSteerings();
  },

  /**
   * Create tooltip element
   */
  createTooltipElement() {
    if (document.getElementById('hatcat-tooltip')) return;

    const tooltip = document.createElement('div');
    tooltip.id = 'hatcat-tooltip';
    tooltip.style.cssText = `
      position: absolute;
      background: #2a2a2a;
      color: #ffffff;
      border: 1px solid #444;
      border-radius: 6px;
      padding: 12px;
      font-size: 12px;
      max-width: 400px;
      z-index: 10000;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      pointer-events: none;
    `;
    document.body.appendChild(tooltip);
    this.activeTooltip = tooltip;
  },

  /**
   * Show tooltip for a token
   */
  showTooltip(element, event) {
    const data = JSON.parse(element.dataset.tooltip);
    const tooltip = this.activeTooltip;

    // Build tooltip HTML
    let html = `<div style="font-weight: bold; margin-bottom: 8px;">
      Token: "${data.token}"
    </div>`;

    // Divergence info
    if (data.max_divergence > 0) {
      html += `<div style="margin-bottom: 8px;">
        <strong>Max Divergence:</strong> ${(data.max_divergence * 100).toFixed(1)}%
      </div>`;

      if (data.top_divergences.length > 0) {
        html += `<div style="margin-bottom: 8px;">
          <strong>Top Divergent Concepts:</strong><br/>`;
        data.top_divergences.forEach(d => {
          html += `<div style="margin-left: 8px; font-size: 11px;">
            • ${d.concept}: ${(d.divergence * 100).toFixed(1)}%
            (act: ${(d.activation * 100).toFixed(0)}%,
             txt: ${(d.text * 100).toFixed(0)}%)
          </div>`;
        });
        html += `</div>`;
      }
    }

    // Palette swatches
    if (data.palette.length > 0) {
      html += `<div style="margin-top: 8px;">
        <strong>Concept Palette:</strong><br/>
        <div style="display: flex; gap: 4px; margin-top: 4px;">`;
      data.palette.forEach(p => {
        const textColor = getContrastingTextColor(p.color);
        html += `<div style="
          background-color: ${p.color};
          color: ${textColor};
          padding: 4px 8px;
          border-radius: 3px;
          font-size: 10px;
          flex: 1;
          text-align: center;
        ">${p.concept}</div>`;
      });
      html += `</div></div>`;
    }

    // Steering info
    if (data.steering.active) {
      html += `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444;">
        <strong>⚡ Active Steerings:</strong><br/>`;
      data.steering.steerings.forEach(s => {
        const sign = s.strength > 0 ? '+' : '';
        html += `<div style="margin-left: 8px; font-size: 11px;">
          • ${s.concept}: ${sign}${(s.strength * 100).toFixed(0)}%
          (layer ${s.layer}, ${s.source})
        </div>`;
      });
      html += `</div>`;
    }

    html += `<div style="margin-top: 8px; font-size: 10px; color: #888;">
      Right-click to add steering
    </div>`;

    tooltip.innerHTML = html;

    // Position tooltip
    const rect = element.getBoundingClientRect();
    const tooltipHeight = 300; // Estimate
    const spaceAbove = rect.top;
    const spaceBelow = window.innerHeight - rect.bottom;

    let top, left;
    if (spaceBelow > tooltipHeight || spaceBelow > spaceAbove) {
      // Show below
      top = rect.bottom + window.scrollY + 8;
    } else {
      // Show above
      top = rect.top + window.scrollY - tooltipHeight - 8;
    }

    left = rect.left + window.scrollX;

    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
    tooltip.style.display = 'block';
  },

  /**
   * Hide tooltip
   */
  hideTooltip() {
    if (this.activeTooltip) {
      this.activeTooltip.style.display = 'none';
    }
  },

  /**
   * Create steering context menu
   */
  createSteeringMenu() {
    if (document.getElementById('hatcat-steering-menu')) return;

    const menu = document.createElement('div');
    menu.id = 'hatcat-steering-menu';
    menu.style.cssText = `
      position: absolute;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 6px;
      padding: 8px;
      z-index: 10001;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      min-width: 200px;
    `;
    document.body.appendChild(menu);
    this.steeringMenu = menu;

    // Close on click outside
    document.addEventListener('click', () => {
      menu.style.display = 'none';
    });
  },

  /**
   * Show steering context menu
   */
  showSteeringMenu(element, event) {
    event.preventDefault();
    const data = JSON.parse(element.dataset.tooltip);
    const menu = this.steeringMenu;

    // Build menu for top concepts
    let html = `<div style="color: #fff; font-weight: bold; margin-bottom: 8px; font-size: 13px;">
      Add Steering
    </div>`;

    const concepts = data.top_divergences.slice(0, 3);
    if (concepts.length === 0) {
      html += `<div style="color: #888; font-size: 12px;">No concepts detected</div>`;
    } else {
      concepts.forEach(c => {
        html += `<div class="steering-menu-item" style="
          padding: 6px;
          margin: 4px 0;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
          color: #fff;
        " onmouseover="this.style.background='#444'"
           onmouseout="this.style.background='transparent'">
          <div style="font-weight: bold;">${c.concept}</div>
          <div style="display: flex; gap: 8px; margin-top: 4px;">
            <button onclick="HatCat.addSteering('${c.concept}', 0.5)"
              style="flex: 1; padding: 4px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">
              Amplify
            </button>
            <button onclick="HatCat.addSteering('${c.concept}', -0.5)"
              style="flex: 1; padding: 4px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer;">
              Suppress
            </button>
          </div>
        </div>`;
      });
    }

    menu.innerHTML = html;

    // Position menu
    menu.style.top = `${event.pageY}px`;
    menu.style.left = `${event.pageX}px`;
    menu.style.display = 'block';
  },

  /**
   * Add a steering via API
   */
  async addSteering(concept, strength) {
    try {
      const response = await fetch(`${HATCAT_API_BASE}/steering/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          concept: concept,
          layer: 0,
          strength: strength,
          source: 'user',
          reason: strength > 0 ? 'Amplify from UI' : 'Suppress from UI'
        })
      });

      if (response.ok) {
        console.log(`✅ Added steering: ${concept} (${strength})`);
        this.loadSteerings();
        this.steeringMenu.style.display = 'none';
      } else {
        console.error('Failed to add steering:', await response.text());
      }
    } catch (error) {
      console.error('Error adding steering:', error);
    }
  },

  /**
   * Remove a steering via API
   */
  async removeSteering(concept) {
    try {
      const response = await fetch(
        `${HATCAT_API_BASE}/steering/remove/${encodeURIComponent(concept)}?session_id=${this.sessionId}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        console.log(`✅ Removed steering: ${concept}`);
        this.loadSteerings();
      } else {
        console.error('Failed to remove steering:', await response.text());
      }
    } catch (error) {
      console.error('Error removing steering:', error);
    }
  },

  /**
   * Load active steerings from API
   */
  async loadSteerings() {
    try {
      const response = await fetch(`${HATCAT_API_BASE}/steering/list?session_id=${this.sessionId}`);
      const data = await response.json();
      this.steerings = data.steerings || [];
      this.updateSteeringPanel();
    } catch (error) {
      console.error('Error loading steerings:', error);
    }
  },

  /**
   * Create steering panel UI
   */
  createSteeringPanel() {
    if (document.getElementById('hatcat-steering-panel')) return;

    const panel = document.createElement('div');
    panel.id = 'hatcat-steering-panel';
    panel.style.cssText = `
      position: fixed;
      right: 20px;
      top: 80px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 8px;
      padding: 16px;
      min-width: 250px;
      max-width: 350px;
      z-index: 1000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      color: #fff;
      font-size: 13px;
    `;
    document.body.appendChild(panel);
  },

  /**
   * Update steering panel content
   */
  updateSteeringPanel() {
    const panel = document.getElementById('hatcat-steering-panel');
    if (!panel) return;

    let html = `<div style="font-weight: bold; font-size: 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
      <span>⚡ Active Steerings (${this.steerings.length})</span>
      ${this.steerings.length > 0 ? `<button onclick="HatCat.clearAllSteerings()"
        style="padding: 4px 8px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
        Clear All
      </button>` : ''}
    </div>`;

    if (this.steerings.length === 0) {
      html += `<div style="color: #888; font-style: italic;">
        No active steerings.<br/>
        Right-click on tokens to add.
      </div>`;
    } else {
      this.steerings.forEach(s => {
        const strengthPercent = (s.strength * 100).toFixed(0);
        const sign = s.strength > 0 ? '+' : '';
        const color = s.strength > 0 ? '#4CAF50' : '#f44336';

        html += `<div style="
          margin: 8px 0;
          padding: 10px;
          background: #333;
          border-radius: 6px;
          border-left: 3px solid ${color};
        ">
          <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
              <div style="font-weight: bold; margin-bottom: 4px;">${s.concept}</div>
              <div style="font-size: 11px; color: #aaa;">
                Layer ${s.layer} • ${s.source}
              </div>
              <div style="margin-top: 6px; background: #222; border-radius: 3px; height: 6px; overflow: hidden;">
                <div style="background: ${color}; height: 100%; width: ${Math.abs(s.strength) * 100}%;"></div>
              </div>
              <div style="font-size: 10px; color: #aaa; margin-top: 2px;">
                ${sign}${strengthPercent}%
              </div>
            </div>
            <button onclick="HatCat.removeSteering('${s.concept}')"
              style="padding: 4px 8px; background: transparent; color: #f44336; border: 1px solid #f44336; border-radius: 4px; cursor: pointer; font-size: 11px; margin-left: 8px;">
              Remove
            </button>
          </div>
        </div>`;
      });
    }

    panel.innerHTML = html;
  },

  /**
   * Clear all steerings
   */
  async clearAllSteerings() {
    try {
      const response = await fetch(
        `${HATCAT_API_BASE}/steering/clear?session_id=${this.sessionId}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        console.log('✅ Cleared all steerings');
        this.loadSteerings();
      }
    } catch (error) {
      console.error('Error clearing steerings:', error);
    }
  }
};

// ============================================================================
// OpenWebUI Integration
// ============================================================================

/**
 * Process streaming chunk from OpenWebUI
 * @param {object} chunk - SSE chunk from API
 * @returns {string} Processed HTML content
 */
function processStreamChunk(chunk) {
  if (!chunk.choices || !chunk.choices[0]) return '';

  const delta = chunk.choices[0].delta;
  if (!delta || !delta.content) return '';

  const token = delta.content;
  const metadata = delta.metadata;

  return renderToken(token, metadata);
}

// Initialize on load
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    HatCat.init();
  });
}

// Export for OpenWebUI
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { HatCat, renderToken, processStreamChunk };
}
