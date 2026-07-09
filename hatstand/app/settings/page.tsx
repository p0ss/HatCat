"use client";

import { PageHeader } from "@/components/page-header";
import {
  Badge,
  Card,
  CardBody,
  CardHeader,
  CardTitle,
  ErrorState,
} from "@/components/ui";
import { useEnv } from "@/lib/hooks/use-env";
import { useHealth } from "@/lib/hooks/use-health";

function humanizeBytes(bytes: number | undefined): string {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return `${value.toFixed(value >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-4 py-2 text-sm border-b border-zinc-100 last:border-0 dark:border-zinc-800">
      <div className="text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className="text-zinc-900 dark:text-zinc-100 font-mono text-xs break-all">
        {children}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const env = useEnv();
  const health = useHealth();

  return (
    <div className="px-8 py-6">
      <PageHeader
        title="Settings"
        description="Backend connection, env detection, defaults."
      />

      <div className="mt-6 grid gap-4 grid-cols-1 lg:grid-cols-2">
        {/* ---- Backend ---- */}
        <Card>
          <CardHeader>
            <CardTitle>Backend</CardTitle>
          </CardHeader>
          <CardBody>
            <Row label="HatCatDev URL">
              {env.data?.hatcatdev_url ?? "—"}
            </Row>
            <Row label="Connection">
              {health.isLoading ? (
                <Badge variant="muted">checking…</Badge>
              ) : health.isError ? (
                <Badge variant="error">unreachable</Badge>
              ) : (
                <Badge variant="success">connected</Badge>
              )}
            </Row>
            <Row label="Version">
              {health.data?.version ?? "—"}
            </Row>
            <Row label="Uptime">
              {health.data
                ? `${Math.round(health.data.uptime_seconds)}s`
                : "—"}
            </Row>
          </CardBody>
        </Card>

        {/* ---- Auth ---- */}
        <Card>
          <CardHeader>
            <CardTitle>Auth</CardTitle>
          </CardHeader>
          <CardBody>
            <Row label="Admin token">
              {env.isLoading ? (
                <Badge variant="muted">checking…</Badge>
              ) : env.data?.admin_token_configured ? (
                <Badge variant="success">configured</Badge>
              ) : (
                <Badge variant="warning">not set</Badge>
              )}
            </Row>
            <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
              Set <code className="font-mono">HATSTAND_ADMIN_TOKEN</code> in{" "}
              <code className="font-mono">.env.local</code> to enable
              authenticated mutations. The token is never echoed to the client.
            </p>
          </CardBody>
        </Card>

        {/* ---- Environment ---- */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Environment</CardTitle>
          </CardHeader>
          <CardBody>
            {env.isError ? (
              <ErrorState
                title="Could not read environment"
                message={env.error?.message}
                onRetry={() => env.refetch()}
              />
            ) : (
              <>
                <Row label="Python env">
                  {env.data
                    ? env.data.python.active_env === "none"
                      ? "none"
                      : `${env.data.python.active_env}${
                          env.data.python.env_name
                            ? ` (${env.data.python.env_name})`
                            : ""
                        }`
                    : "—"}
                </Row>
                <Row label="Python version">
                  {env.data?.python.version ?? "not detected"}
                </Row>
                <Row label="Node version">
                  {env.data?.node.version ?? "—"}
                </Row>
                <Row label="HF cache path">
                  {env.data?.hf_cache.path ?? "—"}
                </Row>
                <Row label="HF cache size">
                  {humanizeBytes(env.data?.hf_cache.size_bytes)}
                </Row>
                <Row label="GPU">
                  {env.data?.gpu ? (
                    <span>
                      {env.data.gpu.name} —{" "}
                      {humanizeBytes(env.data.gpu.available_memory_bytes)} free
                      / {humanizeBytes(env.data.gpu.total_memory_bytes)} total
                    </span>
                  ) : (
                    "not detected"
                  )}
                </Row>
              </>
            )}
          </CardBody>
        </Card>

        {/* ---- Defaults ---- */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Defaults</CardTitle>
          </CardHeader>
          <CardBody>
            <Row label="Default substrate">
              {env.data?.default_substrate ?? (
                <span className="text-zinc-500 dark:text-zinc-400">
                  not configured
                </span>
              )}
            </Row>
            <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
              Settings storage is not yet wired. Default-substrate selection
              will land alongside settings persistence.
            </p>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
