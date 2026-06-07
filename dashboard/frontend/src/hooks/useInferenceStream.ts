import { useCallback, useEffect, useRef, useState } from "react";
import type { DashboardFrame } from "../types/nexus";

const MAX_QUEUE = 5;

export function useInferenceStream(cameraId: string, wsBase = "") {
  const [frame, setFrame] = useState<DashboardFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queueRef = useRef<DashboardFrame[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const drainQueue = useCallback(() => {
    if (queueRef.current.length > 0) {
      const next = queueRef.current.shift();
      if (next) setFrame(next);
    }
  }, []);

  useEffect(() => {
    const url = `${wsBase}/ws/dashboard/${cameraId}`.replace(/^http/, "ws");
    const ws = new WebSocket(url.replace("https", "wss"));
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as DashboardFrame;
        queueRef.current.push(payload);
        if (queueRef.current.length > MAX_QUEUE) {
          queueRef.current.shift();
        }
        drainQueue();
      } catch {
        setError("Failed to parse dashboard frame");
      }
    };

    ws.onerror = () => setError("WebSocket connection error");
    ws.onclose = () => setConnected(false);

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [cameraId, wsBase, drainQueue]);

  return { frame, connected, error };
}
