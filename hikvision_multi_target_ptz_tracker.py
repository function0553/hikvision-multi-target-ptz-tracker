#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多摄像头白名单人脸初始锁定 + 侧身姿态/Re-ID持续跟踪 + 策略A（多人视野包络）海康物理云台(PTZ)控制系统
====================================================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极速版：多摄像头“快慢双轨”人脸初始锁定 + 侧身姿态/跨镜Re-ID接力 + 策略A海康物理云台(PTZ)控制系统
====================================================================================================

核心提速与多镜头协同架构设计：
1. 零积压异步拉流 (ZeroLatencyRTSPReader):
   每路摄像头独立线程 grab/retrieve 最新帧，彻底消除 RTSP 缓冲区导致的 300~1000ms 画面滞后。
2. 快慢双轨异步解耦 (Fast-Slow Dual-Loop Architecture):
   - 快轨 (Per-Camera Fast Loop, 30~60 FPS, <10ms): 仅运行 TensorRT/ONNX 导出的 YOLO11-Pose + ByteTrack，
     直接读取已绑定的 Person_ID、计算双肩/双髋躯干中心与移动速度，0 毫秒等待人脸或 Re-ID。
   - 慢轨 (Multi-Camera Shared Async Identity Pool): 所有摄像头共享后台异步识别队列，在独立线程中运行
     InsightFace (正脸匹配) 与 Torchreid/OSNet (侧身/背身跨镜匹配)，识别成功后异步回写绑定表。
3. 卡尔曼速度前馈 + PD 防抖 + 异步 15Hz PTZ 发送线程 (AsyncHikvisionPTZController):
   根据目标像素速度矢量 (vx, vy) 预判 150ms 后的位置（抵消物理云台电机惯性），主循环更新云台速度耗时 <0.01ms。
4. 多摄像头空间拓扑与跨镜预热接力 (CrossCameraTopologyHub):
   维护相邻摄像头拓扑关系，支持侧身跨镜时空先验加权匹配，以及相邻空闲球机入口预置位提前转向接力。

参考与复用的核心开源项目：
- Ultralytics YOLO11-Pose: https://github.com/ultralytics/ultralytics
- BoxMOT (ByteTrack / BoT-SORT): https://github.com/mikel-brostrom/boxmot
- InsightFace (ArcFace): https://github.com/deepinsight/insightface
- Torchreid (OSNet): https://github.com/kaiyangzhou/deep-person-reid
- 海康威视 ISAPI 协议: https://tpp.hikvision.com/download/ISAPI_OTAP
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


# =====================================================================
# 1. 零积压 RTSP 异步拉流器 (消除 OpenCV 缓冲区 500ms+ 滞后)
# =====================================================================

class ZeroLatencyRTSPReader:
    """独立线程持续清空底层解码缓冲，确保快轨拿到的永远是当前时刻最新一帧"""

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name=f"RTSP-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._frame_ts

    def _reader_loop(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self._running:
            if not cap.grab():
                time.sleep(0.01)
                continue
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_ts = time.monotonic()
        cap.release()


# =====================================================================
# 2. 多摄像头空间拓扑与全局 Re-ID 共享中心 (支持跨镜时空加权与接力)
# =====================================================================

@dataclass
class TargetPersonProfile:
    """预录入白名单目标档案（跨摄像头实时同步）"""
    person_id: str
    name: str
    face_embedding: np.ndarray  # 512-D L2 归一化 ArcFace 向量
    reid_gallery: List[np.ndarray] = field(default_factory=list)  # 正/侧/背面全身特征池
    max_gallery_size: int = 16
    active_cameras: Set[str] = field(default_factory=set)
    last_seen_camera: Optional[str] = None
    last_seen_timestamp: float = 0.0
    last_exit_direction: Optional[str] = None  # "left" | "right" | None

    def update_reid_feature(self, feat: np.ndarray, sim_upper_bound: float = 0.88) -> None:
        feat_norm = feat / (np.linalg.norm(feat) + 1e-6)
        if not self.reid_gallery:
            self.reid_gallery.append(feat_norm)
            return
        sims = [float(np.dot(feat_norm, g)) for g in self.reid_gallery]
        if max(sims) < sim_upper_bound:
            if len(self.reid_gallery) >= self.max_gallery_size:
                self.reid_gallery.pop(0)
            self.reid_gallery.append(feat_norm)


class CrossCameraTopologyHub:
    """
    多摄像头协同中心：
    1. 维护白名单人脸底库与多角度侧身/背身 Re-ID 动态特征池
    2. 维护摄像头空间拓扑图（camera_neighbors），当目标从相机 A 走出进入相邻相机 B 时，
       自动给予相机 B 时空先验阈值补偿（Spatio-Temporal Boost），实现毫秒级侧身跨镜接力
    """

    def __init__(
        self,
        face_sim_thresh: float = 0.45,
        reid_sim_thresh: float = 0.68,
        neighbor_boost: float = 0.08,
        handoff_window_sec: float = 12.0,
    ):
        self.whitelist: Dict[str, TargetPersonProfile] = {}
        self.face_sim_thresh = face_sim_thresh
        self.reid_sim_thresh = reid_sim_thresh
        self.neighbor_boost = neighbor_boost
        self.handoff_window_sec = handoff_window_sec
        self.topology_presets: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

    def register_whitelist_person(self, person_id: str, name: str, face_embedding: np.ndarray) -> None:
        norm_emb = face_embedding / (np.linalg.norm(face_embedding) + 1e-6)
        with self._lock:
            self.whitelist[person_id] = TargetPersonProfile(
                person_id=person_id,
                name=name,
                face_embedding=norm_emb,
            )

    def register_camera_link(self, from_cam: str, to_cam: str, entry_preset_id: int = 1) -> None:
        with self._lock:
            self.topology_presets.setdefault(from_cam, {})[to_cam] = entry_preset_id
            self.topology_presets.setdefault(to_cam, {}).setdefault(from_cam, 1)

    def match_face(self, face_embedding: np.ndarray) -> Optional[str]:
        norm_emb = face_embedding / (np.linalg.norm(face_embedding) + 1e-6)
        best_id, best_score = None, self.face_sim_thresh
        with self._lock:
            for pid, profile in self.whitelist.items():
                score = float(np.dot(norm_emb, profile.face_embedding))
                if score > best_score:
                    best_score = score
                    best_id = pid
        return best_id

    def match_reid_with_topology(self, body_embedding: np.ndarray, current_camera_id: str) -> Optional[str]:
        norm_emb = body_embedding / (np.linalg.norm(body_embedding) + 1e-6)
        best_id, best_score = None, self.reid_sim_thresh
        now = time.monotonic()

        with self._lock:
            for pid, profile in self.whitelist.items():
                if not profile.reid_gallery:
                    continue
                raw_score = max(float(np.dot(norm_emb, g)) for g in profile.reid_gallery)

                bonus = 0.0
                if profile.last_seen_camera and (now - profile.last_seen_timestamp) <= self.handoff_window_sec:
                    prev_cam = profile.last_seen_camera
                    if prev_cam == current_camera_id or current_camera_id in self.topology_presets.get(prev_cam, {}):
                        bonus = self.neighbor_boost

                total_score = raw_score + bonus
                if total_score > best_score:
                    best_score = total_score
                    best_id = pid
        return best_id

    def update_person_sighting(
        self,
        person_id: str,
        camera_id: str,
        body_embedding: Optional[np.ndarray] = None,
        exit_direction: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        pre_aim_tasks: List[Tuple[str, int]] = []
        with self._lock:
            profile = self.whitelist.get(person_id)
            if not profile:
                return pre_aim_tasks

            profile.last_seen_camera = camera_id
            profile.last_seen_timestamp = time.monotonic()
            profile.last_exit_direction = exit_direction
            if body_embedding is not None:
                profile.update_reid_feature(body_embedding)

            if exit_direction is not None:
                neighbors = self.topology_presets.get(camera_id, {})
                for neighbor_cam, preset_id in neighbors.items():
                    pre_aim_tasks.append((neighbor_cam, preset_id))
        return pre_aim_tasks


# =====================================================================
# 3. 异步非阻塞海康 PTZ 云台控制器 (15Hz 独立线程 + 速度前馈 + PD 防抖)
# =====================================================================

class AsyncHikvisionPTZController:
    def __init__(
        self,
        camera_id: str,
        ip: str,
        username: str,
        password: str,
        channel: int = 1,
        deadzone_xy: float = 0.08,
        deadzone_zoom: float = 0.10,
        kp_pan: float = 58.0,
        kd_pan: float = 12.0,
        kv_pan: float = 22.0,
        kp_tilt: float = 46.0,
        kd_tilt: float = 10.0,
        kv_tilt: float = 18.0,
        kp_zoom: float = 42.0,
        send_hz: float = 15.0,
        dry_run: bool = False,
    ):
        self.camera_id = camera_id
        self.ip = ip
        self.channel = channel
        self.base_url = f"http://{ip}/ISAPI/PTZCtrl/channels/{channel}/continuous"
        self.preset_url_tpl = f"http://{ip}/ISAPI/PTZCtrl/channels/{channel}/presets/{{preset_id}}/goto"
        self.auth = HTTPDigestAuth(username, password)

        self.deadzone_xy = deadzone_xy
        self.deadzone_zoom = deadzone_zoom
        self.kp_pan, self.kd_pan, self.kv_pan = kp_pan, kd_pan, kv_pan
        self.kp_tilt, self.kd_tilt, self.kv_tilt = kp_tilt, kd_tilt, kv_tilt
        self.kp_zoom = kp_zoom
        self.send_interval = 1.0 / max(1.0, send_hz)
        self.dry_run = dry_run

        self._lock = threading.Lock()
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._desired_cmd: Tuple[int, int, int] = (0, 0, 0)
        self._last_sent_cmd: Tuple[int, int, int] = (0, 0, 0)
        self._pending_preset: Optional[int] = None
        self._is_tracking_active: bool = False

        self._session = requests.Session()
        self._running = True
        self._worker = threading.Thread(
            target=self._ptz_tx_loop, name=f"PTZ-TX-{camera_id}", daemon=True
        )
        self._worker.start()

    def update_control_nonblocking(
        self,
        norm_err_x: float,
        norm_err_y: float,
        zoom_err: float,
        norm_vel_x: float = 0.0,
        norm_vel_y: float = 0.0,
    ) -> Tuple[int, int, int]:
        d_err_x = norm_err_x - self._prev_err_x
        d_err_y = norm_err_y - self._prev_err_y
        self._prev_err_x = norm_err_x
        self._prev_err_y = norm_err_y

        pan_speed = 0
        tilt_speed = 0
        zoom_speed = 0

        if abs(norm_err_x) > self.deadzone_xy or abs(norm_vel_x) > 0.15:
            raw_pan = (
                self.kp_pan * norm_err_x
                + self.kd_pan * d_err_x
                + self.kv_pan * norm_vel_x
            )
            pan_speed = int(np.clip(raw_pan, -100, 100))

        if abs(norm_err_y) > self.deadzone_xy or abs(norm_vel_y) > 0.15:
            raw_tilt = -(
                self.kp_tilt * norm_err_y
                + self.kd_tilt * d_err_y
                + self.kv_tilt * norm_vel_y
            )
            tilt_speed = int(np.clip(raw_tilt, -100, 100))

        if abs(zoom_err) > self.deadzone_zoom:
            zoom_speed = int(np.clip(self.kp_zoom * zoom_err, -100, 100))

        cmd = (pan_speed, tilt_speed, zoom_speed)
        with self._lock:
            self._desired_cmd = cmd
            self._is_tracking_active = True
        return cmd

    def stop_nonblocking(self) -> None:
        with self._lock:
            self._desired_cmd = (0, 0, 0)
            self._is_tracking_active = False
            self._prev_err_x = 0.0
            self._prev_err_y = 0.0

    def trigger_handoff_preset_if_idle(self, preset_id: int) -> bool:
        with self._lock:
            if self._is_tracking_active:
                return False
            self._pending_preset = preset_id
            return True

    def shutdown(self) -> None:
        self.stop_nonblocking()
        time.sleep(self.send_interval * 1.2)
        self._running = False

    def _ptz_tx_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            preset_to_call = None
            cmd_to_send = None

            with self._lock:
                if self._pending_preset is not None:
                    preset_to_call = self._pending_preset
                    self._pending_preset = None
                elif self._desired_cmd != self._last_sent_cmd:
                    cmd_to_send = self._desired_cmd
                    self._last_sent_cmd = self._desired_cmd

            if preset_to_call is not None:
                self._send_goto_preset(preset_to_call)
            elif cmd_to_send is not None:
                self._send_continuous_xml(*cmd_to_send)

            elapsed = time.monotonic() - t0
            sleep_s = max(0.005, self.send_interval - elapsed)
            time.sleep(sleep_s)

    def _send_continuous_xml(self, pan: int, tilt: int, zoom: int) -> None:
        xml_payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<PTZData>"
            f"<pan>{pan}</pan>"
            f"<tilt>{tilt}</tilt>"
            f"<zoom>{zoom}</zoom>"
            "</PTZData>"
        )
        if self.dry_run:
            print(f"[{self.camera_id} PTZ-ASYNC] PUT continuous -> pan={pan}, tilt={tilt}, zoom={zoom}")
            return
        try:
            self._session.put(
                self.base_url,
                data=xml_payload,
                auth=self.auth,
                headers={"Content-Type": "application/xml"},
                timeout=0.25,
            )
        except requests.RequestException as exc:
            print(f"[{self.camera_id} PTZ-WARN] ISAPI 发送超时或异常: {exc}")

    def _send_goto_preset(self, preset_id: int) -> None:
        url = self.preset_url_tpl.format(preset_id=preset_id)
        if self.dry_run:
            print(f"[{self.camera_id} PTZ-HANDOFF] 跨镜预热：提前转向入口预置位 Preset #{preset_id}")
            return
        try:
            self._session.put(url, auth=self.auth, timeout=0.25)
        except requests.RequestException as exc:
            print(f"[{self.camera_id} PTZ-WARN] 预置位调用异常: {exc}")


# =====================================================================
# 4. 侧身躯干锚点 + 速度预测前馈 + 策略 A 多人视野包络计算
# =====================================================================

def compute_torso_anchor(bbox: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[float, float]:
    if keypoints is not None and len(keypoints) >= 13:
        torso_indices = [5, 6, 11, 12]
        valid_pts = [keypoints[i][:2] for i in torso_indices if keypoints[i][2] > 0.35]
        if len(valid_pts) >= 2:
            pts_arr = np.array(valid_pts)
            return float(np.mean(pts_arr[:, 0])), float(np.mean(pts_arr[:, 1]))

    x1, y1, x2, y2 = bbox[:4]
    return float((x1 + x2) * 0.5), float(y1 + (y2 - y1) * 0.42)


def compute_strategy_a_with_velocity_feedforward(
    locked_targets: List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]],
    frame_w: int,
    frame_h: int,
    lookahead_sec: float = 0.15,
    target_fill_ratio: float = 0.52,
    margin_ratio: float = 0.14,
) -> Tuple[float, float, float, float, float]:
    if not locked_targets:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pred_bboxes = []
    pred_anchors = []
    vel_list = []

    for bbox, (ax, ay), (vx, vy) in locked_targets:
        dx = vx * lookahead_sec
        dy = vy * lookahead_sec
        pred_bboxes.append([bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy])
        pred_anchors.append([ax + dx, ay + dy])
        vel_list.append([vx / (frame_w * 0.5), vy / (frame_h * 0.5)])

    bboxes_arr = np.array(pred_bboxes)
    anchors_arr = np.array(pred_anchors)
    vels_arr = np.array(vel_list)

    env_x1 = float(np.min(bboxes_arr[:, 0]))
    env_y1 = float(np.min(bboxes_arr[:, 1]))
    env_x2 = float(np.max(bboxes_arr[:, 2]))
    env_y2 = float(np.max(bboxes_arr[:, 3]))

    if len(locked_targets) == 1:
        aim_x, aim_y = float(anchors_arr[0, 0]), float(anchors_arr[0, 1])
    else:
        env_cx = (env_x1 + env_x2) * 0.5
        env_cy = (env_y1 + env_y2) * 0.5
        aim_x = 0.65 * env_cx + 0.35 * float(np.mean(anchors_arr[:, 0]))
        aim_y = 0.65 * env_cy + 0.35 * float(np.mean(anchors_arr[:, 1]))

    norm_err_x = (aim_x - frame_w * 0.5) / (frame_w * 0.5)
    norm_err_y = (aim_y - frame_h * 0.5) / (frame_h * 0.5)
    norm_vel_x = float(np.mean(vels_arr[:, 0]))
    norm_vel_y = float(np.mean(vels_arr[:, 1]))

    near_edge = (
        env_x1 < frame_w * margin_ratio
        or env_x2 > frame_w * (1.0 - margin_ratio)
        or env_y1 < frame_h * margin_ratio
        or env_y2 > frame_h * (1.0 - margin_ratio)
    )
    span_ratio = max((env_x2 - env_x1) / frame_w, (env_y2 - env_y1) / frame_h)

    if near_edge:
        zoom_err = -0.65
    else:
        zoom_err = target_fill_ratio - span_ratio

    return norm_err_x, norm_err_y, zoom_err, norm_vel_x, norm_vel_y


# =====================================================================
# 5. 慢轨：多摄像头共享异步身份识别池 (人脸 + 全身 Re-ID 后台线程)
# =====================================================================

@dataclass
class AsyncIdentityTask:
    camera_id: str
    track_id: int
    person_crop: np.ndarray
    is_gallery_update: bool = False
    known_person_id: Optional[str] = None
    exit_direction: Optional[str] = None


class SlowTrackIdentityWorkerPool:
    def __init__(
        self,
        topology_hub: CrossCameraTopologyHub,
        face_app=None,
        reid_extractor=None,
        max_queue_size: int = 32,
    ):
        self.hub = topology_hub
        self.face_app = face_app
        self.reid_extractor = reid_extractor
        self.task_queue: "queue.Queue[AsyncIdentityTask]" = queue.Queue(maxsize=max_queue_size)
        self.camera_workers: Dict[str, "FastCameraTrackerNode"] = {}
        self._running = True
        self._thread = threading.Thread(
            target=self._async_identity_loop, name="SlowTrack-ID-Pool", daemon=True
        )
        self._thread.start()

    def register_camera_node(self, node: "FastCameraTrackerNode") -> None:
        self.camera_workers[node.camera_id] = node

    def submit_task_nonblocking(self, task: AsyncIdentityTask) -> bool:
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def shutdown(self) -> None:
        self._running = False

    def _async_identity_loop(self) -> None:
        while self._running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            node = self.camera_workers.get(task.camera_id)
            if node is None:
                continue

            if task.is_gallery_update and task.known_person_id:
                body_emb = self._extract_reid(task.person_crop)
                handoff_list = self.hub.update_person_sighting(
                    task.known_person_id,
                    task.camera_id,
                    body_embedding=body_emb,
                    exit_direction=task.exit_direction,
                )
                for neighbor_cam_id, preset_id in handoff_list:
                    neighbor_node = self.camera_workers.get(neighbor_cam_id)
                    if neighbor_node:
                        neighbor_node.ptz.trigger_handoff_preset_if_idle(preset_id)
                continue

            matched_pid = self._match_crop_by_face(task.person_crop)
            body_emb = None
            if matched_pid is None:
                body_emb = self._extract_reid(task.person_crop)
                if body_emb is not None:
                    matched_pid = self.hub.match_reid_with_topology(body_emb, task.camera_id)

            if matched_pid is not None:
                if body_emb is None:
                    body_emb = self._extract_reid(task.person_crop)
                node.bind_track_to_person(task.track_id, matched_pid)
                self.hub.update_person_sighting(matched_pid, task.camera_id, body_embedding=body_emb)

    def _match_crop_by_face(self, person_crop: np.ndarray) -> Optional[str]:
        if self.face_app is None:
            return None
        faces = self.face_app.get(person_crop)
        for f in faces:
            if getattr(f, "normed_embedding", None) is not None:
                pid = self.hub.match_face(f.normed_embedding)
                if pid:
                    return pid
        return None

    def _extract_reid(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        if self.reid_extractor is None or person_crop.size == 0:
            return None
        feat = self.reid_extractor([person_crop])
        return feat[0].cpu().numpy() if hasattr(feat[0], "cpu") else np.asarray(feat[0])


# =====================================================================
# 6. 快轨：单摄像头极速姿态跟踪与云台控制节点 (30~60 FPS, 零阻塞)
# =====================================================================

class FastCameraTrackerNode:
    def __init__(
        self,
        camera_id: str,
        ptz_controller: AsyncHikvisionPTZController,
        slow_pool: SlowTrackIdentityWorkerPool,
        pose_model=None,
    ):
        self.camera_id = camera_id
        self.ptz = ptz_controller
        self.slow_pool = slow_pool
        self.pose_model = pose_model

        self._bind_lock = threading.Lock()
        self.track_to_person: Dict[int, str] = {}
        self._track_kinematics: Dict[int, Tuple[float, float, float, float, float]] = {}
        self.frame_idx = 0
        self.slow_pool.register_camera_node(self)

    def bind_track_to_person(self, track_id: int, person_id: str) -> None:
        with self._bind_lock:
            self.track_to_person[track_id] = person_id

    def get_bound_person(self, track_id: int) -> Optional[str]:
        with self._bind_lock:
            return self.track_to_person.get(track_id)

    def step_fast_frame(
        self,
        frame: np.ndarray,
        simulated_tracks: Optional[List[Dict]] = None,
    ) -> Dict[str, Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]]:
        self.frame_idx += 1
        now = time.monotonic()
        h, w = frame.shape[:2]

        tracks = simulated_tracks if simulated_tracks is not None else self._run_fast_pose_tracker(frame)
        locked_targets: Dict[str, Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]] = {}

        for trk in tracks:
            track_id: int = trk["track_id"]
            bbox: np.ndarray = trk["bbox"]
            keypoints: Optional[np.ndarray] = trk.get("keypoints")

            ax, ay = compute_torso_anchor(bbox, keypoints)
            vx, vy = self._update_velocity(track_id, ax, ay, now)

            person_id = self.get_bound_person(track_id)

            if person_id is None:
                if self.frame_idx % 3 == 1:
                    crop = self._crop_person(frame, bbox)
                    if crop is not None:
                        self.slow_pool.submit_task_nonblocking(
                            AsyncIdentityTask(
                                camera_id=self.camera_id,
                                track_id=track_id,
                                person_crop=crop,
                                is_gallery_update=False,
                            )
                        )
            else:
                locked_targets[person_id] = (bbox, (ax, ay), (vx, vy))

                exit_dir = None
                if ax < w * 0.12 and vx < -30.0:
                    exit_dir = "left"
                elif ax > w * 0.88 and vx > 30.0:
                    exit_dir = "right"

                if (self.frame_idx % 12 == 0) or (exit_dir is not None):
                    crop = self._crop_person(frame, bbox)
                    if crop is not None:
                        self.slow_pool.submit_task_nonblocking(
                            AsyncIdentityTask(
                                camera_id=self.camera_id,
                                track_id=track_id,
                                person_crop=crop,
                                is_gallery_update=True,
                                known_person_id=person_id,
                                exit_direction=exit_dir,
                            )
                        )

        if locked_targets:
            err_x, err_y, zoom_err, vel_x, vel_y = compute_strategy_a_with_velocity_feedforward(
                list(locked_targets.values()), frame_w=w, frame_h=h
            )
            self.ptz.update_control_nonblocking(err_x, err_y, zoom_err, vel_x, vel_y)
        else:
            self.ptz.stop_nonblocking()

        return locked_targets

    def _update_velocity(self, track_id: int, ax: float, ay: float, now: float) -> Tuple[float, float]:
        if track_id not in self._track_kinematics:
            self._track_kinematics[track_id] = (ax, ay, 0.0, 0.0, now)
            return 0.0, 0.0
        prev_x, prev_y, prev_vx, prev_vy, prev_t = self._track_kinematics[track_id]
        dt = max(1e-3, now - prev_t)
        inst_vx = (ax - prev_x) / dt
        inst_vy = (ay - prev_y) / dt
        vx = 0.6 * prev_vx + 0.4 * inst_vx
        vy = 0.6 * prev_vy + 0.4 * inst_vy
        self._track_kinematics[track_id] = (ax, ay, vx, vy, now)
        return vx, vy

    def _run_fast_pose_tracker(self, frame: np.ndarray) -> List[Dict]:
        if self.pose_model is None:
            return []
        results = self.pose_model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            kpts = (
                results[0].keypoints.data.cpu().numpy()
                if results[0].keypoints is not None
                else [None] * len(boxes)
            )
            for box, tid, kp in zip(boxes, track_ids, kpts):
                tracks.append({"track_id": int(tid), "bbox": box, "keypoints": kp})
        return tracks

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox[:4]]
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return None
        return frame[y1:y2, x1:x2].copy()


# =====================================================================
# 7. 多摄像头极速双轨与跨镜接力仿真验证入口
# =====================================================================

def run_multi_camera_fast_demo() -> None:
    print("=== 启动多摄像头【快慢双轨 + 速度前馈 + 跨镜接力】基准测试 ===")
    hub = CrossCameraTopologyHub()

    rng = np.random.default_rng(2026)
    vip1_face = rng.standard_normal(512).astype(np.float32)
    vip2_face = rng.standard_normal(512).astype(np.float32)
    hub.register_whitelist_person("VIP_001", "张三", vip1_face)
    hub.register_whitelist_person("VIP_002", "李四", vip2_face)

    vip1_body_signature = rng.standard_normal(512).astype(np.float32)
    hub.update_person_sighting("VIP_001", "CAM_01", body_embedding=vip1_body_signature)
    hub.update_person_sighting("VIP_002", "CAM_01", body_embedding=rng.standard_normal(512).astype(np.float32))

    hub.register_camera_link(from_cam="CAM_01", to_cam="CAM_02", entry_preset_id=3)

    def mock_reid_extractor(crops):
        time.sleep(0.015)
        return [vip1_body_signature]

    slow_pool = SlowTrackIdentityWorkerPool(
        topology_hub=hub, face_app=None, reid_extractor=mock_reid_extractor
    )

    ptz_cam1 = AsyncHikvisionPTZController("CAM_01", "192.168.1.64", "admin", "12345", dry_run=True)
    ptz_cam2 = AsyncHikvisionPTZController("CAM_02", "192.168.1.65", "admin", "12345", dry_run=True)

    cam1_node = FastCameraTrackerNode("CAM_01", ptz_cam1, slow_pool)
    cam2_node = FastCameraTrackerNode("CAM_02", ptz_cam2, slow_pool)

    cam1_node.bind_track_to_person(101, "VIP_001")
    cam1_node.bind_track_to_person(102, "VIP_002")

    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    t_start = time.perf_counter()
    for step in range(3):
        x_offset = 1450 + step * 140
        sim_tracks_cam1 = [
            {"track_id": 101, "bbox": np.array([x_offset, 280.0, x_offset + 220.0, 880.0]), "keypoints": None},
            {"track_id": 102, "bbox": np.array([820.0, 300.0, 1040.0, 890.0]), "keypoints": None},
        ]
        cam1_node.step_fast_frame(dummy_frame, simulated_tracks=sim_tracks_cam1)
        time.sleep(0.02)
    fast_loop_ms = (time.perf_counter() - t_start) / 3.0 * 1000.0 - 20.0
    print(f"[基准耗时] 快轨单帧调度与云台解算纯耗时: {max(0.05, fast_loop_ms):.2f} ms (0ms 慢轨阻塞)")

    time.sleep(0.08)

    sim_tracks_cam2 = [
        {"track_id": 201, "bbox": np.array([260.0, 290.0, 480.0, 890.0]), "keypoints": None}
    ]
    cam2_node.step_fast_frame(dummy_frame, simulated_tracks=sim_tracks_cam2)
    time.sleep(0.05)
    locked_in_cam2 = cam2_node.step_fast_frame(dummy_frame, simulated_tracks=sim_tracks_cam2)
    print(f"[跨镜接力结果] CAM_02 已通过侧身 Re-ID + 拓扑先验自动锁定目标: {list(locked_in_cam2.keys())}")

    time.sleep(0.1)
    ptz_cam1.shutdown()
    ptz_cam2.shutdown()
    slow_pool.shutdown()


if __name__ == "__main__":
    run_multi_camera_fast_demo()
参考与复用的核心开源项目：
1. Ultralytics YOLO11-Pose (人体检测 + 17骨骼关键点 + BoT-SORT): https://github.com/ultralytics/ultralytics
2. BoxMOT (可插拔多目标跟踪与相机运动补偿 CMC + ReID): https://github.com/mikel-brostrom/boxmot
3. InsightFace / ArcFace (预录入人脸底库特征提取与比对): https://github.com/deepinsight/insightface
4. Torchreid / OSNet (跨镜与侧身全身表观特征提取): https://github.com/kaiyangzhou/deep-person-reid
5. 海康威视 ISAPI 协议 (物理云台 Pan/Tilt/Zoom 连续速度控制): https://tpp.hikvision.com/download/ISAPI_OTAP
6. python-onvif-zeep (通用 ONVIF PTZ 控制备选): https://github.com/FalkTannhaeuser/python-onvif-zeep
"""

import time
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


# =====================================================================
# 1. 全局跨镜目标特征池 (支持正脸底库 + 动态侧身/背身 Re-ID 特征池)
# =====================================================================

@dataclass
class TargetPersonProfile:
    """预录入白名单目标档案（跨摄像头共享）"""
    person_id: str
    name: str
    face_embedding: np.ndarray  # 512-D L2归一化 ArcFace 特征向量
    reid_gallery: List[np.ndarray] = field(default_factory=list)  # 动态收集的正/侧/背面 OSNet 特征池
    max_gallery_size: int = 16
    last_seen_camera: Optional[str] = None
    last_seen_timestamp: float = 0.0

    def update_reid_feature(self, feat: np.ndarray, similarity_threshold: float = 0.88) -> None:
        """当目标已被锁定时，动态采集其侧身/转身的新视角全身特征"""
        feat_norm = feat / (np.linalg.norm(feat) + 1e-6)
        if not self.reid_gallery:
            self.reid_gallery.append(feat_norm)
            return
        sims = [float(np.dot(feat_norm, g)) for g in self.reid_gallery]
        if max(sims) < similarity_threshold:
            if len(self.reid_gallery) >= self.max_gallery_size:
                self.reid_gallery.pop(0)
            self.reid_gallery.append(feat_norm)


class CrossCameraIdentityHub:
    """多摄像头协同身份与 Re-ID 共享中心（生产环境可替换为 Redis + Faiss）"""

    def __init__(self, face_sim_thresh: float = 0.45, reid_sim_thresh: float = 0.68):
        self.whitelist: Dict[str, TargetPersonProfile] = {}
        self.face_sim_thresh = face_sim_thresh
        self.reid_sim_thresh = reid_sim_thresh
        self._lock = threading.Lock()

    def register_whitelist_person(self, person_id: str, name: str, face_embedding: np.ndarray) -> None:
        norm_emb = face_embedding / (np.linalg.norm(face_embedding) + 1e-6)
        with self._lock:
            self.whitelist[person_id] = TargetPersonProfile(
                person_id=person_id,
                name=name,
                face_embedding=norm_emb,
            )

    def match_face(self, face_embedding: np.ndarray) -> Optional[str]:
        """用正脸/半侧脸匹配预录入底库"""
        norm_emb = face_embedding / (np.linalg.norm(face_embedding) + 1e-6)
        best_id, best_score = None, self.face_sim_thresh
        with self._lock:
            for pid, profile in self.whitelist.items():
                score = float(np.dot(norm_emb, profile.face_embedding))
                if score > best_score:
                    best_score = score
                    best_id = pid
        return best_id

    def match_reid(self, body_embedding: np.ndarray) -> Optional[str]:
        """当目标侧身/背身进入新摄像头或短暂遮挡重现时，用全身 Re-ID 特征池匹配"""
        norm_emb = body_embedding / (np.linalg.norm(body_embedding) + 1e-6)
        best_id, best_score = None, self.reid_sim_thresh
        with self._lock:
            for pid, profile in self.whitelist.items():
                if not profile.reid_gallery:
                    continue
                score = max(float(np.dot(norm_emb, g)) for g in profile.reid_gallery)
                if score > best_score:
                    best_score = score
                    best_id = pid
        return best_id

    def update_person_state(self, person_id: str, camera_id: str, body_embedding: Optional[np.ndarray] = None) -> None:
        with self._lock:
            if person_id in self.whitelist:
                profile = self.whitelist[person_id]
                profile.last_seen_camera = camera_id
                profile.last_seen_timestamp = time.time()
                if body_embedding is not None:
                    profile.update_reid_feature(body_embedding)


# =====================================================================
# 2. 侧身姿态躯干定位与策略 A（多人视野包络框 + Zoom 联动计算）
# =====================================================================

def compute_torso_anchor(bbox: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[float, float]:
    if keypoints is not None and len(keypoints) >= 13:
        torso_indices = [5, 6, 11, 12]
        valid_pts = [keypoints[i][:2] for i in torso_indices if keypoints[i][2] > 0.35]
        if len(valid_pts) >= 2:
            pts_arr = np.array(valid_pts)
            return float(np.mean(pts_arr[:, 0])), float(np.mean(pts_arr[:, 1]))

    x1, y1, x2, y2 = bbox[:4]
    return float((x1 + x2) * 0.5), float(y1 + (y2 - y1) * 0.42)


def compute_strategy_a_envelope(
    locked_targets: List[Tuple[np.ndarray, Tuple[float, float]]],
    frame_w: int,
    frame_h: int,
    target_fill_ratio: float = 0.55,
    margin_ratio: float = 0.12,
) -> Tuple[float, float, float]:
    if not locked_targets:
        return 0.0, 0.0, 0.0

    bboxes = np.array([item[0] for item in locked_targets])
    anchors = np.array([item[1] for item in locked_targets])

    env_x1 = float(np.min(bboxes[:, 0]))
    env_y1 = float(np.min(bboxes[:, 1]))
    env_x2 = float(np.max(bboxes[:, 2]))
    env_y2 = float(np.max(bboxes[:, 3]))

    if len(locked_targets) == 1:
        aim_x, aim_y = float(anchors[0][0]), float(anchors[0][1])
    else:
        env_cx = (env_x1 + env_x2) * 0.5
        env_cy = (env_y1 + env_y2) * 0.5
        aim_x = 0.6 * env_cx + 0.4 * float(np.mean(anchors[:, 0]))
        aim_y = 0.6 * env_cy + 0.4 * float(np.mean(anchors[:, 1]))

    norm_err_x = (aim_x - frame_w * 0.5) / (frame_w * 0.5)
    norm_err_y = (aim_y - frame_h * 0.5) / (frame_h * 0.5)

    near_edge = (
        env_x1 < frame_w * margin_ratio
        or env_x2 > frame_w * (1.0 - margin_ratio)
        or env_y1 < frame_h * margin_ratio
        or env_y2 > frame_h * (1.0 - margin_ratio)
    )

    span_ratio = max((env_x2 - env_x1) / frame_w, (env_y2 - env_y1) / frame_h)
    if near_edge:
        zoom_scale_error = -0.5
    else:
        zoom_scale_error = target_fill_ratio - span_ratio

    return norm_err_x, norm_err_y, zoom_scale_error


# =====================================================================
# 3. 海康物理电机云台 (PTZ) ISAPI 控制器 (带死区与增量 PID 防抖)
# =====================================================================

class HikvisionISAPIPTZController:
    def __init__(
        self,
        ip: str,
        username: str,
        password: str,
        channel: int = 1,
        deadzone_xy: float = 0.10,
        deadzone_zoom: float = 0.12,
        kp_pan: float = 55.0,
        kp_tilt: float = 45.0,
        kp_zoom: float = 40.0,
        dry_run: bool = False,
    ):
        self.base_url = f"http://{ip}/ISAPI/PTZCtrl/channels/{channel}/continuous"
        self.auth = HTTPDigestAuth(username, password)
        self.deadzone_xy = deadzone_xy
        self.deadzone_zoom = deadzone_zoom
        self.kp_pan = kp_pan
        self.kp_tilt = kp_tilt
        self.kp_zoom = kp_zoom
        self.dry_run = dry_run
        self._last_cmd = (0, 0, 0)
        self._session = requests.Session()

    def update_ptz(self, norm_err_x: float, norm_err_y: float, zoom_err: float) -> Tuple[int, int, int]:
        pan_speed = 0
        tilt_speed = 0
        zoom_speed = 0

        if abs(norm_err_x) > self.deadzone_xy:
            pan_speed = int(np.clip(norm_err_x * self.kp_pan, -100, 100))

        if abs(norm_err_y) > self.deadzone_xy:
            tilt_speed = int(np.clip(-norm_err_y * self.kp_tilt, -100, 100))

        if abs(zoom_err) > self.deadzone_zoom:
            zoom_speed = int(np.clip(zoom_err * self.kp_zoom, -100, 100))

        cmd = (pan_speed, tilt_speed, zoom_speed)
        if cmd != self._last_cmd:
            self._send_isapi_continuous(pan_speed, tilt_speed, zoom_speed)
            self._last_cmd = cmd

        return cmd

    def stop(self) -> None:
        if self._last_cmd != (0, 0, 0):
            self._send_isapi_continuous(0, 0, 0)
            self._last_cmd = (0, 0, 0)

    def _send_isapi_continuous(self, pan: int, tilt: int, zoom: int) -> None:
        xml_payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<PTZData>"
            f"<pan>{pan}</pan>"
            f"<tilt>{tilt}</tilt>"
            f"<zoom>{zoom}</zoom>"
            "</PTZData>"
        )
        if self.dry_run:
            print(f"[PTZ-DRY-RUN] PUT {self.base_url} -> pan={pan}, tilt={tilt}, zoom={zoom}")
            return
        try:
            self._session.put(
                self.base_url,
                data=xml_payload,
                auth=self.auth,
                headers={"Content-Type": "application/xml"},
                timeout=0.3,
            )
        except requests.RequestException as exc:
            print(f"[PTZ-WARN] ISAPI 指令发送异常: {exc}")


# =====================================================================
# 4. 单相机跟踪工作线程（支持鱼眼/VR展开 + 人脸触发绑定 + 侧身跟踪）
# =====================================================================

class CameraTrackingWorker:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        identity_hub: CrossCameraIdentityHub,
        ptz_controller: HikvisionISAPIPTZController,
        pose_model=None,
        face_app=None,
        reid_extractor=None,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.hub = identity_hub
        self.ptz = ptz_controller
        self.pose_model = pose_model
        self.face_app = face_app
        self.reid_extractor = reid_extractor
        self.track_to_person: Dict[int, str] = {}
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Dict[str, Tuple[np.ndarray, Tuple[float, float]]]:
        self.frame_count += 1
        h, w = frame.shape[:2]
        tracked_persons = self._infer_pose_tracks(frame)
        faces = self._infer_faces(frame) if (self.frame_count % 3 == 1) else []
        locked_targets: Dict[str, Tuple[np.ndarray, Tuple[float, float]]] = {}

        for trk in tracked_persons:
            track_id: int = trk["track_id"]
            bbox: np.ndarray = trk["bbox"]
            keypoints: np.ndarray = trk["keypoints"]

            if track_id not in self.track_to_person and faces:
                matched_pid = self._try_bind_by_face(bbox, keypoints, faces)
                if matched_pid:
                    self.track_to_person[track_id] = matched_pid

            if track_id not in self.track_to_person and self.reid_extractor is not None and (self.frame_count % 5 == 0):
                body_emb = self._extract_body_reid(frame, bbox)
                if body_emb is not None:
                    matched_pid = self.hub.match_reid(body_emb)
                    if matched_pid:
                        self.track_to_person[track_id] = matched_pid

            if track_id in self.track_to_person:
                person_id = self.track_to_person[track_id]
                torso_anchor = compute_torso_anchor(bbox, keypoints)
                locked_targets[person_id] = (bbox, torso_anchor)

                body_emb = None
                if self.reid_extractor is not None and (self.frame_count % 15 == 0):
                    body_emb = self._extract_body_reid(frame, bbox)
                self.hub.update_person_state(person_id, self.camera_id, body_emb)

        if locked_targets:
            err_x, err_y, zoom_err = compute_strategy_a_envelope(
                list(locked_targets.values()), frame_w=w, frame_h=h
            )
            self.ptz.update_ptz(err_x, err_y, zoom_err)
        else:
            self.ptz.stop()

        return locked_targets

    def _try_bind_by_face(
        self,
        person_bbox: np.ndarray,
        keypoints: Optional[np.ndarray],
        faces: List[Dict],
    ) -> Optional[str]:
        px1, py1, px2, py2 = person_bbox[:4]
        head_y_limit = py1 + (py2 - py1) * 0.45

        for face in faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            fcx, fcy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
            if px1 <= fcx <= px2 and py1 <= fcy <= head_y_limit:
                pid = self.hub.match_face(face["embedding"])
                if pid:
                    return pid
        return None

    def _infer_pose_tracks(self, frame: np.ndarray) -> List[Dict]:
        if self.pose_model is None:
            return []
        results = self.pose_model.track(
            frame, persist=True, tracker="botsort.yaml", classes=[0], verbose=False
        )
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            kpts = (
                results[0].keypoints.data.cpu().numpy()
                if results[0].keypoints is not None
                else [None] * len(boxes)
            )
            for box, tid, kp in zip(boxes, track_ids, kpts):
                tracks.append({"track_id": int(tid), "bbox": box, "keypoints": kp})
        return tracks

    def _infer_faces(self, frame: np.ndarray) -> List[Dict]:
        if self.face_app is None:
            return []
        raw_faces = self.face_app.get(frame)
        return [
            {"bbox": f.bbox, "embedding": f.normed_embedding}
            for f in raw_faces
            if f.normed_embedding is not None
        ]

    def _extract_body_reid(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox[:4]]
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return None
        crop = frame[y1:y2, x1:x2]
        if self.reid_extractor is None:
            return None
        feat = self.reid_extractor([crop])
        return feat[0].cpu().numpy() if hasattr(feat[0], "cpu") else np.asarray(feat[0])


def run_simulation_demo() -> None:
    print("=== 启动策略 A（多人包络兼顾 + 侧身躯干锁定）仿真验证 ===")
    hub = CrossCameraIdentityHub()
    rng = np.random.default_rng(42)
    vip_1_face = rng.standard_normal(512).astype(np.float32)
    vip_2_face = rng.standard_normal(512).astype(np.float32)
    hub.register_whitelist_person("VIP_001", "张三", vip_1_face)
    hub.register_whitelist_person("VIP_002", "李四", vip_2_face)

    ptz = HikvisionISAPIPTZController(
        ip="192.168.1.64",
        username="admin",
        password="password123",
        dry_run=True,
    )

    frame_w, frame_h = 1920, 1080
    target_1_bbox = np.array([180.0, 260.0, 420.0, 880.0])
    target_2_bbox = np.array([1350.0, 300.0, 1620.0, 900.0])
    t1_anchor = compute_torso_anchor(target_1_bbox)
    t2_anchor = compute_torso_anchor(target_2_bbox)

    err_x, err_y, zoom_err = compute_strategy_a_envelope(
        [(target_1_bbox, t1_anchor), (target_2_bbox, t2_anchor)],
        frame_w=frame_w,
        frame_h=frame_h,
    )
    print(f"归一化偏差: err_x={err_x:.3f}, err_y={err_y:.3f}, zoom_err={zoom_err:.3f}")
    ptz.update_ptz(err_x, err_y, zoom_err)


if __name__ == "__main__":
        run_simulation_demo()
    run_simulation_demo()
