/** Shared TypeScript types for the NEXUS-CV dashboard. */

export interface Detection {
  bbox_xyxy: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
  track_id?: number | null;
}

export interface Track {
  track_id: string;
  state: string;
  age_frames: number;
  modalities_seen: string[];
  last_bbox_2d: [number, number, number, number] | null;
  velocity_2d: [number, number];
  class_votes: Record<string, number>;
  anomaly_score: number;
}

export interface Trajectory {
  track_id: string;
  predicted_positions: [number, number][];
  horizon_frames: number;
  confidence: number;
}

export interface Anomaly {
  track_id: string;
  score: number;
  contributing_factors: string[];
  is_anomalous: boolean;
}

export interface ScenePrediction {
  scene_class: string;
  confidence: number;
  top3: [string, number][];
}

export interface LiveMetrics {
  inference_ms: number;
  active_tracks: number;
  sla_breach_rate: number;
  anomaly_rate: number;
}

export interface DashboardFrame {
  frame_b64: string;
  detections: Detection[];
  tracks: Track[];
  trajectories: Trajectory[];
  anomalies: Anomaly[];
  scene: ScenePrediction;
  metrics: LiveMetrics;
  request_id?: string;
  camera_id?: string;
  timestamp_ns?: number;
  inference_ms?: number;
  serving_ms?: number;
}

export interface ReplaySession {
  session_id: string;
  camera_id: string;
  started_at: number;
  ended_at: number | null;
  frame_count: number;
}

export interface ReplayFrameMeta {
  frame_id: number;
  frame_index: number;
  timestamp_ns: number;
}

export interface AnomalyEvent {
  id: string;
  timestamp_ns: number;
  camera_id: string;
  track_id: string;
  score: number;
  factors: string[];
}
