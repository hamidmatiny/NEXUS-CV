import type { AnomalyEvent } from "../types/nexus";

interface Props {
  events: AnomalyEvent[];
}

function formatTime(ns: number): string {
  return new Date(ns / 1_000_000).toLocaleTimeString();
}

export function AnomalyFeed({ events }: Props) {
  return (
    <div className="rounded-lg bg-nexus-panel border border-slate-700 flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-700 font-semibold text-sm">
        Anomaly Feed
      </div>
      <div className="flex-1 overflow-y-auto max-h-64 p-2 space-y-2">
        {events.length === 0 && (
          <div className="text-nexus-muted text-sm p-2">No anomalies detected</div>
        )}
        {events.map((evt) => (
          <div
            key={evt.id}
            className="rounded border border-red-900/50 bg-red-950/30 p-2 text-sm animate-pulse"
          >
            <div className="flex justify-between items-center">
              <span className="font-mono text-xs text-nexus-muted">
                {formatTime(evt.timestamp_ns)}
              </span>
              <span className="rounded bg-nexus-danger/20 text-nexus-danger px-2 py-0.5 text-xs font-bold">
                {(evt.score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-1">
              <span className="text-nexus-accent">{evt.camera_id}</span> · track{" "}
              {evt.track_id.slice(0, 8)}
            </div>
            <div className="text-xs text-slate-400 mt-1">{evt.factors.join(", ")}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
