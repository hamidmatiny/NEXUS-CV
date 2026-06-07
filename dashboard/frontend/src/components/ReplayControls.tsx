import { useCallback, useEffect, useRef, useState } from "react";
import type { DashboardFrame, ReplayFrameMeta, ReplaySession } from "../types/nexus";

interface Props {
  onFrame: (frame: DashboardFrame | null) => void;
}

export function ReplayControls({ onFrame }: Props) {
  const [sessions, setSessions] = useState<ReplaySession[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [frames, setFrames] = useState<ReplayFrameMeta[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    fetch("/api/v1/replay/sessions")
      .then((r) => r.json())
      .then(setSessions)
      .catch(() => setSessions([]));
  }, []);

  useEffect(() => {
    if (!selectedSession) return;
    fetch(`/api/v1/replay/sessions/${selectedSession}/frames?limit=500`)
      .then((r) => r.json())
      .then((data) => setFrames(data.frames ?? []))
      .catch(() => setFrames([]));
    setFrameIndex(0);
  }, [selectedSession]);

  const loadFrame = useCallback(
    async (idx: number) => {
      const meta = frames[idx];
      if (!meta || !selectedSession) return;
      const res = await fetch(
        `/api/v1/replay/sessions/${selectedSession}/frames/${meta.frame_id}`,
      );
      if (res.ok) {
        const payload = (await res.json()) as DashboardFrame;
        onFrame(payload);
      }
    },
    [frames, selectedSession, onFrame],
  );

  useEffect(() => {
    loadFrame(frameIndex);
  }, [frameIndex, loadFrame]);

  useEffect(() => {
    if (!playing) {
      if (timerRef.current) window.clearInterval(timerRef.current);
      return;
    }
    timerRef.current = window.setInterval(() => {
      setFrameIndex((i) => (i + 1 >= frames.length ? 0 : i + 1));
    }, 33);
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, [playing, frames.length]);

  return (
    <div className="rounded-lg bg-nexus-panel border border-slate-700 p-4 space-y-3">
      <div className="text-sm font-semibold">Session Replay</div>
      <select
        className="w-full rounded bg-slate-900 border border-slate-600 px-3 py-2 text-sm"
        value={selectedSession}
        onChange={(e) => setSelectedSession(e.target.value)}
      >
        <option value="">Select session…</option>
        {sessions.map((s) => (
          <option key={s.session_id} value={s.session_id}>
            {s.camera_id} · {s.frame_count} frames · {new Date(s.started_at * 1000).toLocaleString()}
          </option>
        ))}
      </select>

      {frames.length > 0 && (
        <>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={frameIndex}
            onChange={(e) => setFrameIndex(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex gap-2">
            <button
              className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-sm"
              onClick={() => setFrameIndex((i) => Math.max(0, i - 1))}
            >
              ◀ Step
            </button>
            <button
              className="px-3 py-1 rounded bg-nexus-accent text-slate-900 font-medium text-sm"
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button
              className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-sm"
              onClick={() => setFrameIndex((i) => Math.min(frames.length - 1, i + 1))}
            >
              Step ▶
            </button>
            <span className="text-xs text-nexus-muted self-center ml-auto">
              {frameIndex + 1} / {frames.length}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
