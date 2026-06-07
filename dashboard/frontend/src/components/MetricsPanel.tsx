import { useMetrics } from "../hooks/useMetrics";
import type { LiveMetrics } from "../types/nexus";

interface Props {
  metrics: LiveMetrics | undefined;
  sceneClass?: string;
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return <div className="h-8 bg-slate-800 rounded" />;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox="0 0 100 100" className="h-8 w-full" preserveAspectRatio="none">
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  sparkValues,
  color,
  alert,
}: {
  label: string;
  value: string;
  sparkValues: number[];
  color: string;
  alert?: boolean;
}) {
  return (
    <div
      className={`rounded-lg bg-nexus-panel p-4 border ${
        alert ? "border-nexus-danger" : "border-slate-700"
      }`}
    >
      <div className="text-xs uppercase tracking-wide text-nexus-muted">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      <Sparkline values={sparkValues} color={color} />
    </div>
  );
}

export function MetricsPanel({ metrics, sceneClass }: Props) {
  const { sparkline, latest } = useMetrics(metrics);

  return (
    <div className="grid grid-cols-2 gap-3">
      <MetricCard
        label="Inference (ms)"
        value={latest ? `${latest.inference_ms.toFixed(1)}` : "—"}
        sparkValues={sparkline("inference_ms")}
        color="#22d3ee"
      />
      <MetricCard
        label="Active Tracks"
        value={latest ? `${latest.active_tracks}` : "—"}
        sparkValues={sparkline("active_tracks")}
        color="#a78bfa"
      />
      <MetricCard
        label="SLA Breach Rate"
        value={latest ? `${(latest.sla_breach_rate * 100).toFixed(2)}%` : "—"}
        sparkValues={sparkline("sla_breach_rate")}
        color="#f87171"
        alert={!!latest && latest.sla_breach_rate > 0.01}
      />
      <MetricCard
        label="Anomaly Rate"
        value={latest ? `${(latest.anomaly_rate * 100).toFixed(1)}%` : "—"}
        sparkValues={sparkline("anomaly_rate")}
        color="#fb923c"
      />
      {sceneClass && (
        <div className="col-span-2 rounded-lg bg-nexus-panel p-3 border border-slate-700 text-sm">
          Scene: <span className="text-nexus-accent font-medium">{sceneClass}</span>
        </div>
      )}
    </div>
  );
}
