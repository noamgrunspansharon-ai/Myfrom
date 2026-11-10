"""
pose_engine.py
==============
MediaPipe Pose (BlazePose, 33 landmarks) + OpenCV wrapper.

Two passes keep the pipeline clean:
  1. `extract_poses()`           video  -> PoseSequence (pixel keypoints per frame)
  2. `render_annotated_video()`  video + cleaned PoseSequence + EventAnalysis -> overlay video
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np

from biomechanics import EventAnalysis, PoseSequence, calculate_angle, hip_center

ProgressCB = Optional[Callable[[float, str], None]]

# BGR colours
C_LEFT, C_RIGHT, C_CENTER = (0, 165, 255), (255, 200, 0), (200, 200, 200)
C_ANGLE, C_BANNER, C_TRAIL = (0, 255, 255), (40, 40, 220), (0, 255, 120)


class PoseEngine:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, max_width: int = 1280):
        self.model_complexity = model_complexity
        self.min_det = min_detection_confidence
        self.min_track = min_tracking_confidence
        self.max_width = max_width
        self._mp_pose = mp.solutions.pose
        self._connections = list(self._mp_pose.POSE_CONNECTIONS)

    # ------------------------------------------------------------------ pass 1
    def extract_poses(self, video_path: str, fps_override: float = 0.0, progress_cb: ProgressCB = None) -> PoseSequence:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open the video file.")
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_est = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W, H = self._target_size(w0, h0)

        xy_list, vis_list = [], []
        with self._mp_pose.Pose(static_image_mode=False, model_complexity=self.model_complexity,
                                smooth_landmarks=True, min_detection_confidence=self.min_det,
                                min_tracking_confidence=self.min_track) as pose:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (W, H)) if frame.shape[1] != W or frame.shape[0] != H else frame
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    xy_list.append([[p.x * W, p.y * H] for p in lm])
                    vis_list.append([p.visibility for p in lm])
                else:
                    xy_list.append(np.full((33, 2), np.nan))
                    vis_list.append(np.zeros(33))
                i += 1
                if progress_cb and i % 5 == 0:
                    progress_cb(min(i / n_est, 1.0), f"Pose estimation: frame {i}/{n_est}")
        cap.release()

        if not xy_list:
            raise ValueError("The video contains no readable frames.")
        xy, vis = np.asarray(xy_list, float), np.asarray(vis_list, float)
        if (vis.max(axis=1) > 0).mean() < 0.1:
            raise ValueError("No person was detected. Use a clearer side-on clip with the full body visible.")
        fps = float(fps_override) if fps_override and fps_override > 0 else float(file_fps)
        return PoseSequence(xy, vis, fps, W, H, video_fps=float(file_fps))

    # ------------------------------------------------------------------ pass 2
    def render_annotated_video(self, video_path: str, seq: PoseSequence, out_path: str,
                               analysis: Optional[EventAnalysis] = None, show_trail: bool = True,
                               progress_cb: ProgressCB = None) -> str:
        cap = cv2.VideoCapture(video_path)
        W, H, T = seq.width, seq.height, len(seq)
        fps_out = seq.video_fps or seq.fps
        raw_path = out_path + ".raw.mp4"
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (W, H))
        hold = max(8, int(round(0.6 * fps_out)))
        hips = hip_center(seq)
        scale = H / 720.0
        i = 0
        while i < T:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            self._draw_skeleton(frame, seq.xy[i], seq.vis[i], scale)

            if show_trail:
                pts = np.round(hips[:i + 1]).astype(np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, C_TRAIL, max(1, int(2 * scale)), cv2.LINE_AA)

            y = int(32 * scale)
            if analysis:
                self._label(frame, analysis.event, (12, y), scale, (255, 255, 255))
                for ov in analysis.overlays:
                    if ov.start <= i <= ov.end:
                        y += int(30 * scale)
                        self._label(frame, ov.label, (12, y), scale, C_ANGLE)
                        for a, b, c in ov.triplets:
                            self._draw_angle(frame, seq, i, a, b, c, scale)
                for name, f in analysis.key_frames.items():
                    if f <= i < f + hold:
                        y += int(30 * scale)
                        self._label(frame, f">> {name.upper()}", (12, y), scale, (80, 80, 255))
            self._label(frame, f"t={i / seq.fps:.2f}s  f{i}", (12, H - int(12 * scale)), scale * 0.8, (230, 230, 230))
            writer.write(frame)
            i += 1
            if progress_cb and i % 5 == 0:
                progress_cb(min(i / T, 1.0), f"Rendering overlay: frame {i}/{T}")
        cap.release()
        writer.release()

        if not self._transcode_h264(raw_path, out_path):
            shutil.move(raw_path, out_path)  # fallback (may not play in every browser)
        elif os.path.exists(raw_path):
            os.remove(raw_path)
        return out_path

    @staticmethod
    def read_frame(video_path: str, idx: int) -> Optional[np.ndarray]:
        """Return frame `idx` of a video as an RGB array (for key-frame stills)."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

    # ----------------------------------------------------------------- drawing
    def _draw_skeleton(self, img, pts, vis, scale):
        th = max(2, int(3 * scale))

        def P(j):
            return int(pts[j][0]), int(pts[j][1])

        for a, b in self._connections:
            if (a <= 10 and b <= 10) or vis[a] < 0.3 or vis[b] < 0.3:
                continue
            if a % 2 == 1 and b % 2 == 1:
                col = C_LEFT          # landmarks with odd index belong to the athlete's left side
            elif a % 2 == 0 and b % 2 == 0 and a > 10 and b > 10:
                col = C_RIGHT
            else:
                col = C_CENTER
            cv2.line(img, P(a), P(b), col, th, cv2.LINE_AA)
        for j in range(11, 33):
            if vis[j] >= 0.3:
                cv2.circle(img, P(j), max(3, int(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        if vis[0] >= 0.3:
            cv2.circle(img, P(0), max(4, int(6 * scale)), C_CENTER, -1, cv2.LINE_AA)

    def _draw_angle(self, img, seq, t, a, b, c, scale):
        pa, pb, pc = seq.point(t, a), seq.point(t, b), seq.point(t, c)
        ang = calculate_angle(pa, pb, pc)
        ia, ib, ic = [tuple(int(v) for v in p) for p in (pa, pb, pc)]
        th = max(2, int(4 * scale))
        cv2.line(img, ia, ib, C_ANGLE, th, cv2.LINE_AA)
        cv2.line(img, ib, ic, C_ANGLE, th, cv2.LINE_AA)
        cv2.circle(img, ib, max(5, int(7 * scale)), C_ANGLE, -1, cv2.LINE_AA)
        self._label(img, f"{ang:.0f} deg", (ib[0] + int(10 * scale), ib[1] - int(10 * scale)), scale * 0.9, C_ANGLE)

    @staticmethod
    def _label(img, text, org, scale, color):
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, int(2 * scale))
        cv2.putText(img, text, org, font, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, font, fs, color, th, cv2.LINE_AA)

    # ----------------------------------------------------------------- helpers
    def _target_size(self, w, h):
        s = min(1.0, self.max_width / float(max(w, 1)))
        W, H = int(w * s), int(h * s)
        return W - W % 2, H - H % 2  # even dims for H.264

    @staticmethod
    def _transcode_h264(src: str, dst: str) -> bool:
        """Re-encode mp4v -> H.264 (yuv420p) so Streamlit / browsers can play it."""
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = shutil.which("ffmpeg")
        if not exe:
            return False
        cmd = [exe, "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
               "-crf", "23", "-movflags", "+faststart", "-an", dst]
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
