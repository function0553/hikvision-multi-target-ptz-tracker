#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多摄像头白名单人脸初始锁定 + 侧身姿态/Re-ID持续跟踪 + 策略A（多人视野包络）海康物理云台(PTZ)控制系统
====================================================================================================

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
