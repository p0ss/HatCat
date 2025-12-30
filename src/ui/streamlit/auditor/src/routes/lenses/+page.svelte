<script lang="ts">
	import { ticks, introspection, requestIntrospection } from '$lib/stores/bed';
	import type { ExperienceTick, LensTrace } from '$lib/types';
	import { onMount } from 'svelte';

	// Refresh introspection periodically
	onMount(() => {
		requestIntrospection();
		const interval = setInterval(() => requestIntrospection(), 2000);
		return () => clearInterval(interval);
	});

	// Build lens traces from recent ticks
	let conceptTraces = $derived.by(() => {
		const traces: Record<string, { tick: number; score: number }[]> = {};
		const recentTicks = $ticks.slice(-100);

		for (const tick of recentTicks) {
			for (const [concept, score] of Object.entries(tick.concept_activations)) {
				if (!traces[concept]) traces[concept] = [];
				traces[concept].push({ tick: tick.tick_id, score });
			}
		}

		// Sort by frequency
		return Object.entries(traces)
			.sort((a, b) => b[1].length - a[1].length)
			.slice(0, 20);
	});

	let simplexTraces = $derived.by(() => {
		const traces: Record<string, { tick: number; score: number; deviation: number | null }[]> = {};
		const recentTicks = $ticks.slice(-100);

		for (const tick of recentTicks) {
			for (const [term, score] of Object.entries(tick.simplex_activations)) {
				if (!traces[term]) traces[term] = [];
				traces[term].push({
					tick: tick.tick_id,
					score,
					deviation: tick.simplex_deviations[term] ?? null
				});
			}
		}

		return Object.entries(traces);
	});

	// Mini sparkline renderer
	function sparkline(data: { score: number }[], width: number = 100, height: number = 24): string {
		if (data.length === 0) return '';

		const min = Math.min(...data.map((d) => d.score));
		const max = Math.max(...data.map((d) => d.score));
		const range = max - min || 1;

		const points = data
			.map((d, i) => {
				const x = (i / (data.length - 1 || 1)) * width;
				const y = height - ((d.score - min) / range) * height;
				return `${x},${y}`;
			})
			.join(' ');

		return `M ${points.replace(/ /g, ' L ')}`;
	}

	// Calculate stats
	function stats(data: { score: number }[]): { min: number; max: number; avg: number } {
		if (data.length === 0) return { min: 0, max: 0, avg: 0 };
		const scores = data.map((d) => d.score);
		return {
			min: Math.min(...scores),
			max: Math.max(...scores),
			avg: scores.reduce((a, b) => a + b, 0) / scores.length
		};
	}
</script>

<div class="lenses-page">
	<div class="header">
		<h2>Lens Activations</h2>
		<button onclick={() => requestIntrospection()}>Refresh</button>
	</div>

	<div class="panels">
		<!-- Simplex Lenses -->
		<div class="panel simplex-panel">
			<h3>Simplex Lenses (Proprioception)</h3>

			{#if simplexTraces.length === 0}
				<p class="empty">No simplex data yet</p>
			{:else}
				<div class="lens-list">
					{#each simplexTraces as [term, data]}
						{@const s = stats(data)}
						{@const lastDev = data[data.length - 1]?.deviation}
						<div class="lens-row">
							<div class="lens-info">
								<span class="lens-name">{term}</span>
								<span class="lens-stats">
									{s.avg.toFixed(3)} (±{(s.max - s.min).toFixed(3)})
								</span>
								{#if lastDev !== null && lastDev !== undefined}
									<span
										class="lens-deviation"
										class:warning={Math.abs(lastDev) > 1.5}
										class:danger={Math.abs(lastDev) > 2.5}
									>
										{lastDev > 0 ? '+' : ''}{lastDev.toFixed(2)}σ
									</span>
								{/if}
							</div>
							<svg class="sparkline" width="120" height="28" viewBox="0 0 100 24">
								<path d={sparkline(data)} fill="none" stroke="#58a6ff" stroke-width="1.5" />
							</svg>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Concept Lenses -->
		<div class="panel concept-panel">
			<h3>Concept Activations</h3>

			{#if conceptTraces.length === 0}
				<p class="empty">No concept data yet</p>
			{:else}
				<div class="lens-list">
					{#each conceptTraces as [concept, data]}
						{@const s = stats(data)}
						<div class="lens-row">
							<div class="lens-info">
								<span class="lens-name">{concept}</span>
								<span class="lens-stats">
									avg: {s.avg.toFixed(3)} | {data.length} hits
								</span>
							</div>
							<svg class="sparkline" width="120" height="28" viewBox="0 0 100 24">
								<path d={sparkline(data)} fill="none" stroke="#3fb950" stroke-width="1.5" />
							</svg>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	</div>

	<!-- Introspection Report -->
	{#if $introspection}
		<div class="introspection-panel">
			<h3>Introspection Report</h3>
			<div class="report-grid">
				<span class="label">BE ID:</span>
				<span>{$introspection.be_id}</span>

				<span class="label">XDB ID:</span>
				<span>{$introspection.xdb_id}</span>

				<span class="label">Current Tick:</span>
				<span>#{$introspection.current_tick}</span>

				<span class="label">Workspace:</span>
				<span>{$introspection.workspace_state || '-'}</span>

				<span class="label">Tier:</span>
				<span>T{$introspection.tier ?? '?'}</span>
			</div>

			{#if $introspection.recent_violations.length > 0}
				<div class="violations-section">
					<h4>Recent Violations ({$introspection.recent_violations.length})</h4>
					<div class="violation-list">
						{#each $introspection.recent_violations.slice(-5) as v}
							<div class="violation-item">
								<span class="v-term">{v.simplex_term}</span>
								<span class="v-value">{v.deviation?.toFixed(2)}σ</span>
								<span class="v-threshold">threshold: {v.threshold}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			{#if $introspection.recent_steering.length > 0}
				<div class="steering-section">
					<h4>Recent Steering ({$introspection.recent_steering.length})</h4>
					<div class="steering-list">
						{#each $introspection.recent_steering.slice(-5) as s}
							<div class="steering-item">
								<span class="s-term">{s.simplex_term}</span>
								<span class="s-strength">×{s.strength.toFixed(2)}</span>
								<span class="s-target">{s.target_pole}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.lenses-page {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.header h2 {
		margin: 0;
		font-size: 1.1rem;
		color: #c9d1d9;
	}

	.header button {
		background: #21262d;
		border: 1px solid #30363d;
		color: #c9d1d9;
		padding: 0.25rem 0.75rem;
		border-radius: 4px;
		cursor: pointer;
	}

	.header button:hover {
		background: #30363d;
	}

	.panels {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}

	.panel {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		padding: 1rem;
	}

	.panel h3 {
		margin: 0 0 1rem 0;
		font-size: 0.95rem;
		color: #c9d1d9;
	}

	.empty {
		color: #6e7681;
		font-style: italic;
		font-size: 0.85rem;
	}

	.lens-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.lens-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem;
		background: #0d1117;
		border-radius: 4px;
	}

	.lens-info {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
	}

	.lens-name {
		color: #c9d1d9;
		font-size: 0.85rem;
		font-weight: 500;
	}

	.lens-stats {
		color: #8b949e;
		font-size: 0.75rem;
	}

	.lens-deviation {
		font-size: 0.75rem;
		color: #7ee787;
	}

	.lens-deviation.warning {
		color: #f0883e;
	}

	.lens-deviation.danger {
		color: #f85149;
	}

	.sparkline {
		flex-shrink: 0;
	}

	.introspection-panel {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		padding: 1rem;
	}

	.introspection-panel h3 {
		margin: 0 0 1rem 0;
		font-size: 0.95rem;
		color: #c9d1d9;
	}

	.introspection-panel h4 {
		margin: 1rem 0 0.5rem 0;
		font-size: 0.85rem;
		color: #8b949e;
	}

	.report-grid {
		display: grid;
		grid-template-columns: auto 1fr auto 1fr;
		gap: 0.5rem 1rem;
		font-size: 0.85rem;
	}

	.label {
		color: #8b949e;
	}

	.violation-list,
	.steering-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.violation-item,
	.steering-item {
		display: flex;
		gap: 1rem;
		font-size: 0.8rem;
		padding: 0.25rem 0.5rem;
		background: rgba(248, 81, 73, 0.1);
		border-radius: 4px;
	}

	.steering-item {
		background: rgba(240, 136, 62, 0.1);
	}

	.v-term,
	.s-term {
		color: #c9d1d9;
		flex: 1;
	}

	.v-value {
		color: #f85149;
	}

	.v-threshold {
		color: #8b949e;
	}

	.s-strength {
		color: #f0883e;
	}

	.s-target {
		color: #8b949e;
	}
</style>
