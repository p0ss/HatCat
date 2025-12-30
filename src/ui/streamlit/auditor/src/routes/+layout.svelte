<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { connect, disconnect, connected, bedStatus, connectionError } from '$lib/stores/bed';

	let { children } = $props();

	onMount(() => {
		connect();
	});

	onDestroy(() => {
		disconnect();
	});
</script>

<div class="app">
	<header>
		<h1>BED Auditor</h1>
		<nav>
			<a href="/">Stream</a>
			<a href="/lenses">Lenses</a>
			<a href="/xdb">XDB</a>
			<a href="/audit">Audit</a>
			<a href="/tools">Tools</a>
		</nav>
		<div class="status">
			{#if $connected}
				<span class="connected">Connected</span>
				{#if $bedStatus}
					<span class="be-id">{$bedStatus.be_id}</span>
					<span class="tier">T{$bedStatus.tier ?? '?'}</span>
				{/if}
			{:else}
				<span class="disconnected">Disconnected</span>
				{#if $connectionError}
					<span class="error">{$connectionError}</span>
				{/if}
			{/if}
		</div>
	</header>

	<main>
		{@render children()}
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
		background: #0d1117;
		color: #c9d1d9;
	}

	.app {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	}

	header {
		display: flex;
		align-items: center;
		gap: 2rem;
		padding: 0.75rem 1.5rem;
		background: #161b22;
		border-bottom: 1px solid #30363d;
	}

	h1 {
		margin: 0;
		font-size: 1.25rem;
		font-weight: 600;
		color: #58a6ff;
	}

	nav {
		display: flex;
		gap: 1rem;
	}

	nav a {
		color: #8b949e;
		text-decoration: none;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
		transition: all 0.15s;
	}

	nav a:hover {
		color: #c9d1d9;
		background: #21262d;
	}

	.status {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: 0.85rem;
	}

	.connected {
		color: #3fb950;
	}

	.disconnected {
		color: #f85149;
	}

	.error {
		color: #f85149;
		font-size: 0.75rem;
	}

	.be-id {
		color: #8b949e;
	}

	.tier {
		background: #238636;
		color: white;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
		font-size: 0.75rem;
		font-weight: 600;
	}

	main {
		flex: 1;
		padding: 1rem;
		overflow: auto;
	}
</style>
