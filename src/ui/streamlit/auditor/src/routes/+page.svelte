<script lang="ts">
	import {
		ticks,
		latestTick,
		currentTick,
		recentViolations,
		recentSteering,
		clearTicks,
		clearAlerts,
		bedStatus
	} from '$lib/stores/bed';
	import { audit } from '$lib/api';
	import type { ExperienceTick, AuditRecord, AuditCheckpoint, ConceptActivationSummary } from '$lib/types';

	// View mode: 'stream' for live ticks, 'audit' for CAT audit log
	let viewMode: 'stream' | 'audit' = $state('stream');

	// Audit state
	let auditRecords: AuditRecord[] = $state([]);
	let auditCheckpoints: AuditCheckpoint[] = $state([]);
	let auditLoading = $state(false);
	let catId = $state('default-cat');

	let autoScroll = $state(true);
	let streamContainer: HTMLElement;

	// Concept activation summary (computed from visible data)
	let conceptSummary = $derived.by(() => {
		const data = viewMode === 'stream' ? $ticks : auditRecords;
		const summaries: Record<string, { total: number; max: number; count: number; recent: number[] }> = {};

		for (const item of data.slice(-100)) {
			const activations = viewMode === 'stream'
				? (item as ExperienceTick).concept_activations
				: (item as AuditRecord).lens_activations;

			for (const [concept, score] of Object.entries(activations)) {
				if (!summaries[concept]) {
					summaries[concept] = { total: 0, max: 0, count: 0, recent: [] };
				}
				summaries[concept].total += score;
				summaries[concept].max = Math.max(summaries[concept].max, score);
				summaries[concept].count += 1;
				summaries[concept].recent.push(score);
				if (summaries[concept].recent.length > 10) {
					summaries[concept].recent.shift();
				}
			}
		}

		// Convert to sorted array with trend
		return Object.entries(summaries)
			.map(([concept_id, s]) => {
				const avg = s.total / s.count;
				const recent = s.recent;
				let trend: 'rising' | 'falling' | 'stable' = 'stable';
				if (recent.length >= 3) {
					const firstHalf = recent.slice(0, Math.floor(recent.length / 2));
					const secondHalf = recent.slice(Math.floor(recent.length / 2));
					const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
					const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
					if (secondAvg > firstAvg * 1.1) trend = 'rising';
					else if (secondAvg < firstAvg * 0.9) trend = 'falling';
				}
				return {
					concept_id,
					total_activations: s.total,
					avg_score: avg,
					max_score: s.max,
					tick_count: s.count,
					trend
				} as ConceptActivationSummary;
			})
			.sort((a, b) => b.avg_score - a.avg_score)
			.slice(0, 15);
	});

	// Load audit records
	async function loadAuditRecords() {
		const xdbId = $bedStatus?.xdb_id ?? 'xdb-default';
		auditLoading = true;
		try {
			const result = await audit.getRecords(catId, xdbId, 100);
			auditRecords = result.records ?? [];
		} catch (e) {
			console.error('Failed to load audit records:', e);
		} finally {
			auditLoading = false;
		}
	}

	// Load audit checkpoints
	async function loadAuditCheckpoints() {
		const xdbId = $bedStatus?.xdb_id ?? 'xdb-default';
		try {
			const result = await audit.getCheckpoints(catId, xdbId);
			auditCheckpoints = result.checkpoints ?? [];
		} catch (e) {
			console.error('Failed to load checkpoints:', e);
		}
	}

	// Switch view mode
	function setViewMode(mode: 'stream' | 'audit') {
		viewMode = mode;
		if (mode === 'audit') {
			loadAuditRecords();
			loadAuditCheckpoints();
		}
	}

	// Auto-scroll when new ticks arrive
	$effect(() => {
		if (autoScroll && streamContainer && $ticks.length > 0) {
			streamContainer.scrollTop = streamContainer.scrollHeight;
		}
	});

	function formatTime(timestamp: string): string {
		return new Date(timestamp).toLocaleTimeString('en-US', {
			hour12: false,
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit',
			fractionalSecondDigits: 3
		});
	}

	function getTickTypeClass(type: string): string {
		const classes: Record<string, string> = {
			input: 'type-input',
			output: 'type-output',
			introspect: 'type-introspect',
			steering: 'type-steering',
			workspace: 'type-workspace',
			system: 'type-system'
		};
		return classes[type] || '';
	}

	function topActivations(tick: ExperienceTick): [string, number][] {
		const entries = Object.entries(tick.concept_activations);
		return entries.sort((a, b) => b[1] - a[1]).slice(0, 3);
	}

	function getTrendIcon(trend: 'rising' | 'falling' | 'stable'): string {
		if (trend === 'rising') return '↑';
		if (trend === 'falling') return '↓';
		return '→';
	}
</script>

<div class="stream-page">
	<div class="controls">
		<div class="view-toggle">
			<button
				class:active={viewMode === 'stream'}
				onclick={() => setViewMode('stream')}
			>
				Experience Stream
			</button>
			<button
				class:active={viewMode === 'audit'}
				onclick={() => setViewMode('audit')}
			>
				CAT Audit Log
			</button>
		</div>
		<div class="control-buttons">
			{#if viewMode === 'stream'}
				<label>
					<input type="checkbox" bind:checked={autoScroll} />
					Auto-scroll
				</label>
				<button onclick={() => clearTicks()}>Clear</button>
				<span class="tick-count">{$ticks.length} ticks</span>
			{:else}
				<button onclick={loadAuditRecords}>Refresh</button>
				<span class="tick-count">{auditRecords.length} records</span>
			{/if}
		</div>
	</div>

	<div class="panels">
		<!-- Main stream -->
		<div class="stream-panel" bind:this={streamContainer}>
			{#if viewMode === 'stream'}
				{#each $ticks as tick (tick.tick_id)}
					<div class="tick {getTickTypeClass(tick.tick_type)}">
						<div class="tick-header">
							<span class="tick-id">#{tick.tick_id}</span>
							<span class="tick-time">{formatTime(tick.timestamp)}</span>
							<span class="tick-type">{tick.tick_type}</span>
							{#if tick.tier !== null}
								<span class="tick-tier">T{tick.tier}</span>
							{/if}
						</div>
						<div class="tick-content">
							{#if tick.content}
								<span class="content-text">{tick.content}</span>
							{/if}
						</div>
						{#if Object.keys(tick.concept_activations).length > 0}
							<div class="tick-activations">
								{#each topActivations(tick) as [concept, score]}
									<span class="activation" style="opacity: {0.3 + score * 0.7}">
										{concept}: {score.toFixed(2)}
									</span>
								{/each}
							</div>
						{/if}
						{#if tick.hush_violations.length > 0}
							<div class="tick-violations">
								{#each tick.hush_violations as v}
									<span class="violation">! {v.simplex_term}</span>
								{/each}
							</div>
						{/if}
					</div>
				{/each}

				{#if $ticks.length === 0}
					<div class="empty-state">
						<p>Waiting for experience stream...</p>
						<p class="hint">Generate text to see ticks appear here.</p>
					</div>
				{/if}
			{:else}
				<!-- Audit Log View -->
				{#if auditLoading}
					<div class="empty-state">
						<p>Loading audit records...</p>
					</div>
				{:else if auditRecords.length === 0}
					<div class="empty-state">
						<p>No audit records found</p>
						<p class="hint">Audit records are created during BE inference.</p>
					</div>
				{:else}
					{#each auditRecords as record (record.id)}
						<div class="tick {getTickTypeClass(record.event_type)}">
							<div class="tick-header">
								<span class="tick-id">#{record.tick}</span>
								<span class="tick-time">{formatTime(record.timestamp)}</span>
								<span class="tick-type">{record.event_type}</span>
								<span class="record-hash" title={record.record_hash}>
									#{record.record_hash.slice(0, 8)}
								</span>
							</div>
							<div class="tick-content">
								{#if record.raw_content}
									<span class="content-text">{record.raw_content}</span>
								{/if}
							</div>
							{#if Object.keys(record.lens_activations).length > 0}
								<div class="tick-activations">
									{#each Object.entries(record.lens_activations).slice(0, 5) as [concept, score]}
										<span class="activation">
											{concept}: {score.toFixed(2)}
										</span>
									{/each}
								</div>
							{/if}
							{#if record.steering_applied.length > 0}
								<div class="tick-steering">
									{#each record.steering_applied as s}
										<span class="steering-tag">⟳ {s.simplex_term}</span>
									{/each}
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			{/if}
		</div>

		<!-- Sidebar -->
		<div class="sidebar">
			<!-- Concept Activation Summary -->
			<div class="panel concept-summary">
				<h3>Concept Summary</h3>
				{#if conceptSummary.length === 0}
					<p class="empty">No concept data yet</p>
				{:else}
					<div class="concept-list">
						{#each conceptSummary as cs}
							<div class="concept-row">
								<span class="concept-name">{cs.concept_id}</span>
								<span class="concept-stats">
									<span class="avg">{cs.avg_score.toFixed(2)}</span>
									<span class="trend" class:rising={cs.trend === 'rising'} class:falling={cs.trend === 'falling'}>
										{getTrendIcon(cs.trend)}
									</span>
									<span class="count">({cs.tick_count})</span>
								</span>
							</div>
						{/each}
					</div>
				{/if}
			</div>

			{#if viewMode === 'stream'}
				<!-- Latest tick detail -->
				<div class="panel latest-tick">
					<h3>Latest Tick</h3>
					{#if $latestTick}
						<div class="detail-grid">
							<span class="label">ID:</span>
							<span>#{$latestTick.tick_id}</span>

							<span class="label">Type:</span>
							<span>{$latestTick.tick_type}</span>

							<span class="label">State:</span>
							<span>{$latestTick.workspace_state || '-'}</span>

							<span class="label">Tier:</span>
							<span>{$latestTick.tier ?? '-'}</span>

							<span class="label">H-Norm:</span>
							<span>{$latestTick.hidden_state_norm?.toFixed(2) || '-'}</span>
						</div>

						{#if Object.keys($latestTick.simplex_activations).length > 0}
							<h4>Simplex</h4>
							<div class="simplex-list">
								{#each Object.entries($latestTick.simplex_activations) as [term, value]}
									{@const dev = $latestTick.simplex_deviations[term]}
									<div class="simplex-item">
										<span class="term">{term}</span>
										<span class="value">{value.toFixed(3)}</span>
										{#if dev !== null && dev !== undefined}
											<span class="deviation" class:warning={Math.abs(dev) > 1.5}>
												{dev > 0 ? '+' : ''}{dev.toFixed(2)}σ
											</span>
										{/if}
									</div>
								{/each}
							</div>
						{/if}
					{:else}
						<p class="empty">No ticks yet</p>
					{/if}
				</div>

				<!-- Alerts -->
				<div class="panel alerts">
					<h3>
						Alerts
						{#if $recentViolations.length > 0}
							<span class="badge">{$recentViolations.length}</span>
						{/if}
					</h3>

					{#if $recentViolations.length > 0}
						<div class="alert-list">
							{#each $recentViolations.slice(-5) as v}
								<div class="alert violation">
									<span class="alert-type">VIOLATION</span>
									<span class="alert-term">{v.simplex_term}</span>
									<span class="alert-value">{v.deviation?.toFixed(2)}σ</span>
								</div>
							{/each}
						</div>
					{/if}

					{#if $recentSteering.length > 0}
						<div class="alert-list">
							{#each $recentSteering.slice(-5) as s}
								<div class="alert steering">
									<span class="alert-type">STEERING</span>
									<span class="alert-term">{s.simplex_term}</span>
									<span class="alert-value">×{s.strength.toFixed(2)}</span>
								</div>
							{/each}
						</div>
					{/if}

					{#if $recentViolations.length === 0 && $recentSteering.length === 0}
						<p class="empty">No alerts</p>
					{/if}

					{#if $recentViolations.length > 0 || $recentSteering.length > 0}
						<button class="clear-btn" onclick={() => clearAlerts()}>Clear</button>
					{/if}
				</div>
			{:else}
				<!-- Checkpoint History (Audit Mode) -->
				<div class="panel checkpoints">
					<h3>Checkpoints ({auditCheckpoints.length})</h3>
					{#if auditCheckpoints.length === 0}
						<p class="empty">No checkpoints</p>
					{:else}
						<div class="checkpoint-list">
							{#each auditCheckpoints.slice(0, 10) as cp}
								<div class="checkpoint-item">
									<div class="cp-header">
										<span class="cp-range">#{cp.start_tick}-#{cp.end_tick}</span>
										<span class="cp-count">{cp.record_count} records</span>
									</div>
									<div class="cp-time">{formatTime(cp.timestamp)}</div>
									{#if Object.keys(cp.top_k_activations).length > 0}
										<div class="cp-concepts">
											{#each Object.entries(cp.top_k_activations).slice(0, 3) as [concept, score]}
												<span class="cp-concept">{concept}: {score.toFixed(2)}</span>
											{/each}
										</div>
									{/if}
									{#if cp.steering_count > 0}
										<div class="cp-steering">{cp.steering_count} steering events</div>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.stream-page {
		display: flex;
		flex-direction: column;
		height: calc(100vh - 60px);
		gap: 1rem;
	}

	.controls {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.view-toggle {
		display: flex;
		gap: 0;
		background: #21262d;
		border-radius: 6px;
		padding: 2px;
	}

	.view-toggle button {
		background: transparent;
		border: none;
		color: #8b949e;
		padding: 0.375rem 0.75rem;
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.85rem;
		transition: all 0.15s;
	}

	.view-toggle button:hover {
		color: #c9d1d9;
	}

	.view-toggle button.active {
		background: #30363d;
		color: #c9d1d9;
	}

	.control-buttons {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.control-buttons label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: #8b949e;
		font-size: 0.85rem;
	}

	.control-buttons button {
		background: #21262d;
		border: 1px solid #30363d;
		color: #c9d1d9;
		padding: 0.25rem 0.75rem;
		border-radius: 4px;
		cursor: pointer;
	}

	.control-buttons button:hover {
		background: #30363d;
	}

	.tick-count {
		color: #8b949e;
		font-size: 0.85rem;
	}

	.panels {
		display: grid;
		grid-template-columns: 1fr 300px;
		gap: 1rem;
		flex: 1;
		min-height: 0;
	}

	.stream-panel {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		overflow-y: auto;
		padding: 0.5rem;
	}

	.tick {
		padding: 0.5rem;
		border-bottom: 1px solid #21262d;
		font-size: 0.85rem;
	}

	.tick:last-child {
		border-bottom: none;
	}

	.tick-header {
		display: flex;
		gap: 0.75rem;
		margin-bottom: 0.25rem;
	}

	.tick-id {
		color: #8b949e;
		font-weight: 500;
	}

	.tick-time {
		color: #6e7681;
	}

	.tick-type {
		color: #58a6ff;
		text-transform: uppercase;
		font-size: 0.75rem;
	}

	.tick-tier {
		background: #238636;
		color: white;
		padding: 0 0.375rem;
		border-radius: 3px;
		font-size: 0.7rem;
		font-weight: 600;
	}

	.record-hash {
		color: #6e7681;
		font-family: monospace;
		font-size: 0.7rem;
		margin-left: auto;
	}

	.tick-content {
		color: #c9d1d9;
	}

	.content-text {
		white-space: pre-wrap;
		word-break: break-word;
	}

	.type-input .tick-type {
		color: #a371f7;
	}

	.type-output .tick-type {
		color: #3fb950;
	}

	.type-steering .tick-type {
		color: #f0883e;
	}

	.type-introspect .tick-type {
		color: #58a6ff;
	}

	.tick-activations {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.25rem;
		flex-wrap: wrap;
	}

	.activation {
		font-size: 0.75rem;
		color: #7ee787;
		background: rgba(126, 231, 135, 0.1);
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
	}

	.tick-violations {
		margin-top: 0.25rem;
	}

	.violation {
		font-size: 0.75rem;
		color: #f85149;
		background: rgba(248, 81, 73, 0.1);
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
	}

	.tick-steering {
		margin-top: 0.25rem;
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
	}

	.steering-tag {
		font-size: 0.75rem;
		color: #f0883e;
		background: rgba(240, 136, 62, 0.1);
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: #8b949e;
	}

	.empty-state .hint {
		font-size: 0.85rem;
		color: #6e7681;
	}

	.sidebar {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		overflow-y: auto;
	}

	.panel {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		padding: 0.75rem;
	}

	.panel h3 {
		margin: 0 0 0.75rem 0;
		font-size: 0.9rem;
		color: #c9d1d9;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.panel h4 {
		margin: 0.75rem 0 0.5rem 0;
		font-size: 0.8rem;
		color: #8b949e;
	}

	.badge {
		background: #f85149;
		color: white;
		padding: 0.125rem 0.375rem;
		border-radius: 10px;
		font-size: 0.7rem;
	}

	/* Concept Summary */
	.concept-summary {
		max-height: 280px;
		overflow-y: auto;
	}

	.concept-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.concept-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.25rem 0.375rem;
		background: #0d1117;
		border-radius: 3px;
		font-size: 0.8rem;
	}

	.concept-name {
		color: #c9d1d9;
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.concept-stats {
		display: flex;
		gap: 0.375rem;
		align-items: center;
	}

	.concept-stats .avg {
		color: #7ee787;
		font-weight: 500;
	}

	.concept-stats .trend {
		color: #8b949e;
		font-size: 0.75rem;
	}

	.concept-stats .trend.rising {
		color: #3fb950;
	}

	.concept-stats .trend.falling {
		color: #f85149;
	}

	.concept-stats .count {
		color: #6e7681;
		font-size: 0.7rem;
	}

	.detail-grid {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.25rem 0.75rem;
		font-size: 0.8rem;
	}

	.label {
		color: #8b949e;
	}

	.simplex-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.simplex-item {
		display: flex;
		gap: 0.5rem;
		font-size: 0.8rem;
	}

	.simplex-item .term {
		color: #8b949e;
		flex: 1;
	}

	.simplex-item .value {
		color: #c9d1d9;
	}

	.simplex-item .deviation {
		color: #7ee787;
	}

	.simplex-item .deviation.warning {
		color: #f0883e;
	}

	.empty {
		color: #6e7681;
		font-size: 0.8rem;
		font-style: italic;
	}

	.alert-list {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}

	.alert {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		font-size: 0.75rem;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
	}

	.alert.violation {
		background: rgba(248, 81, 73, 0.1);
	}

	.alert.steering {
		background: rgba(240, 136, 62, 0.1);
	}

	.alert-type {
		font-weight: 600;
		font-size: 0.65rem;
	}

	.alert.violation .alert-type {
		color: #f85149;
	}

	.alert.steering .alert-type {
		color: #f0883e;
	}

	.alert-term {
		flex: 1;
		color: #c9d1d9;
	}

	.alert-value {
		color: #8b949e;
	}

	.clear-btn {
		margin-top: 0.5rem;
		width: 100%;
		background: #21262d;
		border: 1px solid #30363d;
		color: #8b949e;
		padding: 0.25rem;
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.75rem;
	}

	.clear-btn:hover {
		background: #30363d;
		color: #c9d1d9;
	}

	/* Checkpoint styles */
	.checkpoint-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.checkpoint-item {
		background: #0d1117;
		border-radius: 4px;
		padding: 0.5rem;
		font-size: 0.8rem;
	}

	.cp-header {
		display: flex;
		justify-content: space-between;
		margin-bottom: 0.25rem;
	}

	.cp-range {
		color: #c9d1d9;
		font-weight: 500;
	}

	.cp-count {
		color: #8b949e;
		font-size: 0.75rem;
	}

	.cp-time {
		color: #6e7681;
		font-size: 0.75rem;
		margin-bottom: 0.25rem;
	}

	.cp-concepts {
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
	}

	.cp-concept {
		font-size: 0.7rem;
		color: #7ee787;
		background: rgba(126, 231, 135, 0.1);
		padding: 0.125rem 0.25rem;
		border-radius: 2px;
	}

	.cp-steering {
		font-size: 0.7rem;
		color: #f0883e;
		margin-top: 0.25rem;
	}
</style>
