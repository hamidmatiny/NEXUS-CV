import { useEffect, useState } from "react";
import type { LiveMetrics } from "../types/nexus";

const WINDOW_MS = 60_000;

export interface MetricSample {
  ts: number;
  inference_ms: number;
  active_tracks: number;
  sla_breach_rate: number;
  anomaly_rate: number;
}

export function useMetrics(metrics: LiveMetrics | undefined) {
  const [history, setHistory] = useState<MetricSample[]>([]);

  useEffect(() => {
    if (!metrics) return;
    const now = Date.now();
    setHistory((prev) => {
      const next = [
        ...prev,
        {
          ts: now,
          inference_ms: metrics.inference_ms,
          active_tracks: metrics.active_tracks,
          sla_breach_rate: metrics.sla_breach_rate,
          anomaly_rate: metrics.anomaly_rate,
        },
      ].filter((s) => now - s.ts <= WINDOW_MS);
      return next;
    });
  }, [metrics]);

  const sparkline = (key: keyof Omit<MetricSample, "ts">) =>
    history.map((s) => s[key] as number);

  const latest = history[history.length - 1];

  return { history, sparkline, latest };
}
