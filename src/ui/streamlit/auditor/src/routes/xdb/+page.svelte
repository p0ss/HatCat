<script lang="ts">
	import { bedStatus } from '$lib/stores/bed';
	import { xdb } from '$lib/api';
	import type { XDBTimestep, XDBTag } from '$lib/types';

	let xdbId = $derived($bedStatus?.xdb_id ?? 'xdb-default');

	// State
	let timesteps: XDBTimestep[] = $state([]);
	let tags: XDBTag[] = $state([]);
	let loading = $state(false);
	let error = $state<string | null>(null);

	// Query state
	let searchText = $state('');
	let selectedTagType = $state('');
	let queryLimit = $state(100);

	// Load recent timesteps
	async function loadRecent() {
		loading = true;
		error = null;
		try {
			const result = await xdb.getRecent(xdbId, queryLimit);
			timesteps = result.timesteps ?? [];
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	// Load tags
	async function loadTags() {
		try {
			const result = await xdb.getTags(xdbId, selectedTagType || undefined);
			tags = result.tags ?? [];
		} catch (e) {
			console.error('Failed to load tags:', e);
		}
	}

	// Search
	async function search() {
		if (!searchText.trim()) {
			loadRecent();
			return;
		}

		loading = true;
		error = null;
		try {
			const result = await xdb.query(xdbId, {
				text_search: searchText,
				limit: queryLimit
			});
			timesteps = result.timesteps ?? [];
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	// Format timestamp
	function formatTime(ts: string): string {
		return new Date(ts).toLocaleString();
	}

	// Initial load
	$effect(() => {
		if (xdbId) {
			loadRecent();
			loadTags();
		}
	});
</script>

<div class="xdb-page">
	<div class="header">
		<h2>XDB Browser</h2>
		<span class="xdb-id">{xdbId}</span>
	</div>

	<!-- Search/Filter Bar -->
	<div class="search-bar">
		<input
			type="text"
			placeholder="Search content..."
			bind:value={searchText}
			onkeydown={(e) => e.key === 'Enter' && search()}
		/>
		<button onclick={search}>Search</button>
		<button onclick={loadRecent}>Recent</button>

		<select bind:value={queryLimit}>
			<option value={50}>50</option>
			<option value={100}>100</option>
			<option value={500}>500</option>
		</select>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<div class="content">
		<!-- Timesteps List -->
		<div class="timesteps-panel">
			<h3>Timesteps ({timesteps.length})</h3>

			{#if loading}
				<p class="loading">Loading...</p>
			{:else if timesteps.length === 0}
				<p class="empty">No timesteps found</p>
			{:else}
				<div class="timestep-list">
					{#each timesteps as ts}
						<div class="timestep-item">
							<div class="ts-header">
								<span class="ts-tick">#{ts.tick}</span>
								<span class="ts-type">{ts.event_type}</span>
								<span class="ts-time">{formatTime(ts.timestamp)}</span>
								<span class="ts-fidelity">{ts.fidelity}</span>
							</div>
							<div class="ts-content">
								{ts.content || '(empty)'}
							</div>
							{#if Object.keys(ts.concept_activations || {}).length > 0}
								<div class="ts-concepts">
									{#each Object.entries(ts.concept_activations).slice(0, 5) as [concept, score]}
										<span class="concept-tag">{concept}: {score.toFixed(2)}</span>
									{/each}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Tags Sidebar -->
		<div class="tags-panel">
			<h3>Tags ({tags.length})</h3>

			<div class="tag-filters">
				<select bind:value={selectedTagType} onchange={loadTags}>
					<option value="">All types</option>
					<option value="concept">Concept</option>
					<option value="entity">Entity</option>
					<option value="bud">Bud</option>
					<option value="custom">Custom</option>
				</select>
			</div>

			{#if tags.length === 0}
				<p class="empty">No tags found</p>
			{:else}
				<div class="tag-list">
					{#each tags as tag}
						<div class="tag-item" class:bud={tag.tag_type === 'bud'}>
							<span class="tag-name">{tag.name}</span>
							<span class="tag-type">{tag.tag_type}</span>
							<span class="tag-count">{tag.use_count} uses</span>
							{#if tag.bud_status}
								<span class="bud-status">{tag.bud_status}</span>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.xdb-page {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		height: calc(100vh - 80px);
	}

	.header {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.header h2 {
		margin: 0;
		font-size: 1.1rem;
		color: #c9d1d9;
	}

	.xdb-id {
		color: #8b949e;
		font-size: 0.85rem;
		background: #21262d;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
	}

	.search-bar {
		display: flex;
		gap: 0.5rem;
	}

	.search-bar input {
		flex: 1;
		background: #0d1117;
		border: 1px solid #30363d;
		color: #c9d1d9;
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-family: inherit;
	}

	.search-bar input:focus {
		outline: none;
		border-color: #58a6ff;
	}

	.search-bar button,
	.search-bar select {
		background: #21262d;
		border: 1px solid #30363d;
		color: #c9d1d9;
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		cursor: pointer;
	}

	.search-bar button:hover,
	.search-bar select:hover {
		background: #30363d;
	}

	.error-banner {
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid #f85149;
		color: #f85149;
		padding: 0.5rem 1rem;
		border-radius: 4px;
	}

	.content {
		display: grid;
		grid-template-columns: 1fr 280px;
		gap: 1rem;
		flex: 1;
		min-height: 0;
	}

	.timesteps-panel,
	.tags-panel {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		padding: 1rem;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	h3 {
		margin: 0 0 0.75rem 0;
		font-size: 0.9rem;
		color: #c9d1d9;
	}

	.timestep-list,
	.tag-list {
		overflow-y: auto;
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.timestep-item {
		background: #0d1117;
		border-radius: 4px;
		padding: 0.5rem 0.75rem;
		font-size: 0.85rem;
	}

	.ts-header {
		display: flex;
		gap: 0.75rem;
		margin-bottom: 0.25rem;
	}

	.ts-tick {
		color: #8b949e;
		font-weight: 500;
	}

	.ts-type {
		color: #58a6ff;
		text-transform: uppercase;
		font-size: 0.75rem;
	}

	.ts-time {
		color: #6e7681;
		font-size: 0.75rem;
	}

	.ts-fidelity {
		margin-left: auto;
		font-size: 0.7rem;
		text-transform: uppercase;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
		background: #21262d;
		color: #8b949e;
	}

	.ts-content {
		color: #c9d1d9;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.ts-concepts {
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
		margin-top: 0.25rem;
	}

	.concept-tag {
		font-size: 0.7rem;
		background: rgba(126, 231, 135, 0.1);
		color: #7ee787;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
	}

	.tag-filters {
		margin-bottom: 0.75rem;
	}

	.tag-filters select {
		width: 100%;
		background: #0d1117;
		border: 1px solid #30363d;
		color: #c9d1d9;
		padding: 0.375rem;
		border-radius: 4px;
	}

	.tag-item {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: center;
		padding: 0.5rem;
		background: #0d1117;
		border-radius: 4px;
		font-size: 0.8rem;
	}

	.tag-item.bud {
		border-left: 3px solid #a371f7;
	}

	.tag-name {
		color: #c9d1d9;
		flex: 1;
	}

	.tag-type {
		color: #58a6ff;
		font-size: 0.7rem;
		text-transform: uppercase;
	}

	.tag-count {
		color: #8b949e;
		font-size: 0.7rem;
	}

	.bud-status {
		font-size: 0.65rem;
		background: #a371f7;
		color: white;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
	}

	.empty,
	.loading {
		color: #6e7681;
		font-style: italic;
		font-size: 0.85rem;
	}
</style>
