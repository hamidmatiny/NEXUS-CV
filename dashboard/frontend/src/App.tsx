import { useMemo, useState } from "react";
import { AnomalyFeed } from "./components/AnomalyFeed";
import { MetricsPanel } from "./components/MetricsPanel";
import { ReplayControls } from "./components/ReplayControls";
import { TrackOverlay } from "./components/TrackOverlay";
import { VideoCanvas } from "./components/VideoCanvas";
import { useInferenceStream } from "./hooks/useInferenceStream";
import type { AnomalyEvent, DashboardFrame } from "./types/nexus";

export default function App() {
  const [cameraId] = useState("cam_00");
  const [replayFrame, setReplayFrame] = useState<DashboardFrame | null>(null);
  const { frame: liveFrame, connected, error } = useInferenceStream(cameraId);

  const activeFrame = replayFrame ?? liveFrame;

  const anomalyEvents = useMemo<AnomalyEvent[]>(() => {
    if (!activeFrame) return [];
    return activeFrame.anomalies
      .filter((a) => a.is_anomalous)
      .map((a, i) => ({
        id: `${a.track_id}-${i}`,
        timestamp_ns: activeFrame.timestamp_ns ?? Date.now() * 1_000_000,
        camera_id: activeFrame.camera_id ?? cameraId,
        track_id: a.track_id,
        score: a.score,
        factors: a.contributing_factors,
      }));
  }, [activeFrame, cameraId]);

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">NEXUS-CV</h1>
          <p className="text-nexus-muted text-sm">Live Observability Dashboard</p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span
            className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-400"}`}
          />
          {connected ? "Live" : "Disconnected"}
          {error && <span className="text-nexus-danger ml-2">{error}</span>}
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6 max-w-[1400px]">
        <div className="space-y-4">
          <div className="relative inline-block w-full">
            <VideoCanvas frameB64={activeFrame?.frame_b64 ?? null} />
            {activeFrame && (
              <TrackOverlay
                tracks={activeFrame.tracks}
                trajectories={activeFrame.trajectories}
                anomalies={activeFrame.anomalies}
              />
            )}
          </div>
          <ReplayControls onFrame={setReplayFrame} />
        </div>

        <div className="space-y-4">
          <MetricsPanel
            metrics={activeFrame?.metrics}
            sceneClass={activeFrame?.scene?.scene_class}
          />
          <AnomalyFeed events={anomalyEvents} />
        </div>
      </div>
    </div>
  );
}
