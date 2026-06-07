import type { Anomaly, Track, Trajectory } from "../types/nexus";

interface Props {
  tracks: Track[];
  trajectories: Trajectory[];
  anomalies: Anomaly[];
  width?: number;
  height?: number;
}

function trackColor(trackId: string): string {
  let hash = 0;
  for (let i = 0; i < trackId.length; i++) {
    hash = trackId.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 55%)`;
}

function isAnomalous(trackId: string, anomalies: Anomaly[]): boolean {
  return anomalies.some((a) => a.track_id === trackId && a.is_anomalous);
}

export function TrackOverlay({
  tracks,
  trajectories,
  anomalies,
  width = 960,
  height = 540,
}: Props) {
  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
    >
      {tracks.map((track) => {
        if (!track.last_bbox_2d) return null;
        const [x1, y1, x2, y2] = track.last_bbox_2d;
        const color = trackColor(track.track_id);
        const anomalous = isAnomalous(track.track_id, anomalies);
        const [vx, vy] = track.velocity_2d;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;

        return (
          <g key={track.track_id}>
            <rect
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill="none"
              stroke={color}
              strokeWidth={2}
              className={anomalous ? "animate-pulse" : undefined}
            />
            <text x={x1} y={y1 - 4} fill={color} fontSize={11} fontFamily="monospace">
              {track.track_id.slice(0, 8)}
            </text>
            <line
              x1={cx}
              y1={cy}
              x2={cx + vx * 20}
              y2={cy + vy * 20}
              stroke={color}
              strokeWidth={2}
              markerEnd="url(#arrow)"
            />
            {anomalous && (
              <circle cx={x2} cy={y1} r={6} fill="#f87171" className="animate-pulse" />
            )}
          </g>
        );
      })}

      {trajectories.map((traj) => {
        const points = traj.predicted_positions;
        if (points.length < 2) return null;
        const d = points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
        return (
          <path
            key={traj.track_id}
            d={d}
            fill="none"
            stroke="#94a3b8"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
        );
      })}

      <defs>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  );
}
