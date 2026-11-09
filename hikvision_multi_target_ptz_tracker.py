#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多品牌通用 (适配天云伟业/公版ONVIF/大华/海康) & 有界整帧全景 + 非360°限位云台跟踪系统
====================================================================================================

针对【天云伟业 + 整帧全景 + 不支持360°旋转（带机械限位）】的专项设计：
1. 移除错误的 360° 环绕假设 (`BoundedPanoramaProcessor`):
   - 摄像头不支持 360° 旋转，其“整帧全景”是有界广角/全景画面（左右边缘为真实物理边界，不可首尾相连）。
   - 移除 `cv2.BORDER_WRAP` 与 360° 四扇区环绕，改为整帧直接检测 + 可选径向畸变矫正 (`cv2.undistort`) + 有界边界处理 (`cv2.BORDER_CONSTANT`)。
2. 非 360° 机械限位保护与防堵转跨镜接力 (`LimitAwarePTZController`):
   - 针对不支持 360° 连续旋转的云台（存在左右/上下物理止挡限位），内置软限位行程积分器 (`max_pan_travel_sec`) 与边缘到达判定；
   - 当云台转到左/右机械限位尽头且目标仍在往外走时，立即切断同向 Pan 电机指令（防止电机顶死限位器打齿发热），
     同时强制拉广角 (Zoom Out) 并向相邻摄像头发送跨镜接力信号。
3. 天云伟业等国产球机协议体检与自动探测 (`diagnose_camera_compatibility` / `--diagnose-ip`):
   - 自动扫描摄像头开放端口 (80, 8899, 8080, 5000, 8000, 554, 34567 等)，探测是否支持标准 ONVIF Profile S 或公版 NetSDK，
     自动反查真实 RTSP 流地址并实测云台点动响应，3 秒内给出“能否直接控制”的明确结论。
"""

import argparse
import glob
import math
import os
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


# =====================================================================
# 1. 天云伟业 / 未知品牌球机硬件协议体检器 (--diagnose-ip)
# =====================================================================

COMMON_CAMERA_PORTS = {
    80: "HTTP Web / 标准 ONVIF 端口",
    8899: "国内公版方案 (天云伟业/雄迈常见) ONVIF 端口",
    8080: "备用 HTTP / ONVIF 端口",
    5000: "部分国产 IPC ONVIF 端口",
    8000: "海康 SDK / 备用 ONVIF 端口",
    554: "标准 RTSP 视频流端口",
    8554: "备用 RTSP 视频流端口",
    34567: "雄迈 (XMEye/NetSDK) 私有协议TCP端口",
    37777: "大华私有协议TCP端口",
}


def scan_open_ports(ip: str, timeout: float = 0.6) -> Dict[int, str]:
    """扫描目标摄像头开启的 TCP 端口，判断其底层主板方案"""
    open_ports: Dict[int, str] = {}
    for port, desc in COMMON_CAMERA_PORTS.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            if sock.connect_ex((ip, port)) == 0:
                open_ports[port] = desc
    return open_ports


def diagnose_camera_compatibility(ip: str, username: str, password: str) -> None:
    """
    针对天云伟业等摄像头的实机协议体检：
    1. 扫描常见控制与视频端口
    2. 轮询所有开放端口尝试 ONVIF 握手、读取设备信息、提取 RTSP 地址、测试云台点动
    """
    print(f"=== 开始体检摄像头 [{ip}] 的控制协议与视频流兼容性 ===")
    open_ports = scan_open_ports(ip)
    if not open_ports:
        print(f"[结论: 无法连接] 未在 {ip} 扫描到任何常见摄像头端口，请检查 IP 是否正确或网络是否互通。")
        return

    print("1. 已开放端口列表:")
    for p, desc in open_ports.items():
        print(f"   - 端口 {p}: {desc}")

    onvif_candidates = [p for p in (80, 8899, 8080, 5000, 8000) if p in open_ports] + [80, 8899, 8080, 5000]
    seen = set()
    onvif_ports = [p for p in onvif_candidates if not (p in seen or seen.add(p))]

    for port in onvif_ports:
        try:
            driver = ONVIFPTZDriver(ip, port, username, password)
            rtsp_uri = driver.get_rtsp_stream_uri()
            print(f"\n[结论: 可直接控制!] 在端口 {port} 成功握手 ONVIF 云台服务！")
            print(f"   - 自动获取到的 RTSP 视频流地址: {rtsp_uri or '未返回，请使用端口 554 手动指定'}")
            print("   - 正在执行 0.3 秒云台微动自检...")
            driver.move_continuous(0.2, 0.0, 0.0)
            time.sleep(0.3)
            driver.stop()
            print("   - 云台微动指令发送成功！")
            return
        except Exception as exc:
            print(f"   - 尝试端口 {port} ONVIF 握手未通过: {exc}")

    if 34567 in open_ports:
        print(
            "\n[结论: 需开启 ONVIF 或使用 NetSDK] 检测到端口 34567 开放（常见于天云伟业采用的公版/雄迈主板方案）。\n"
            "请登录摄像头网页后台（或使用 ONVIF Device Manager），在【网络设置 -> ONVIF】中勾选启用 ONVIF（通常在 8899 或 80 端口），即可直接被本系统控制。"
        )
    else:
        print(
            "\n[结论: 未探测到标准 ONVIF 响应] 请检查相机后台是否已开启 ONVIF 功能及账号密码是否正确。"
        )


# =====================================================================
# 2. 有界整帧全景处理器 (无 360° 环绕，适配半球广角 / 180° 整帧全景)
# =====================================================================

class BoundedPanoramaProcessor:
    """
    针对“整帧全景但摄像头不支持 360° 旋转”的有界广角/半球画面处理器：
    - 绝不使用 `cv2.BORDER_WRAP`（不会将画面左边缘与右边缘跨界拼接）；
    - 支持直接对整帧全景画面进行全局人体检测（视野内 100% 无盲区，无需轮询扇区）；
    - 支持可选的广角/半球径向畸变矫正 (`k1, k2, k3`)，矫正边缘人物侧身弯曲变形。
    """

    def __init__(
        self,
        enable_undistort: bool = False,
        dist_coeffs: Tuple[float, float, float, float] = (-0.22, 0.05, 0.0, 0.0),
    ):
        self.enable_undistort = enable_undistort
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self._map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def preprocess_full_frame(self, frame: np.ndarray) -> np.ndarray:
        """对整帧全景画面进行有界畸变矫正（若未开启矫正，则原样返回整帧全景图直接做全局检测）"""
        if not self.enable_undistort:
            return frame

        h, w = frame.shape[:2]
        key = (w, h)
        if key not in self._map_cache:
            fx, fy = w * 0.75, w * 0.75
            cx, cy = w * 0.5, h * 0.5
            cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, self.dist_coeffs, (w, h), alpha=0.5)
            map1, map2 = cv2.initUndistortRectifyMap(
                cam_mtx, self.dist_coeffs, None, new_mtx, (w, h), cv2.CV_32FC1
            )
            self._map_cache[key] = (map1, map2)

        map1, map2 = self._map_cache[key]
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# =====================================================================
# 3. 多厂商通用云台驱动层 (ONVIF / 大华 CGI / 海康 ISAPI)
# =====================================================================

class BasePTZDriver(ABC):
    @abstractmethod
    def move_continuous(self, pan_norm: float, tilt_norm: float, zoom_norm: float) -> None:
        """归一化连续速度控制: pan_norm, tilt_norm, zoom_norm in [-1.0, 1.0]"""

    @abstractmethod
    def stop(self) -> None:
        """立即停止云台"""

    @abstractmethod
    def goto_preset(self, preset_id: int) -> None:
        """转到指定预置位"""


class ONVIFPTZDriver(BasePTZDriver):
    """
    通用 ONVIF Profile S 物理云台驱动（兼容天云伟业、宇视、大华、海康等所有开启 ONVIF 的球机）：
    - 自动轮询 80 / 8899 / 8080 / 5000 / 8000 端口
    - 支持通过 ONVIF `GetStreamUri` 自动查询真实 RTSP 地址
    - 内置 PanTilt+Zoom 联合控制向纯 PanTilt 控制的容错降级
    """

    def __init__(self, ip: str, port: int, username: str, password: str):
        from onvif import ONVIFCamera

        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.cam = ONVIFCamera(ip, port, username, password)
        self.media = self.cam.create_media_service()
        self.ptz = self.cam.create_ptz_service()
        profiles = self.media.GetProfiles()
        if not profiles:
            raise RuntimeError(f"ONVIF 相机 {ip}:{port} 未返回任何 Media Profile")
        self.profile_token = profiles[0].token
        self._supports_joint_zoom = True

    def get_rtsp_stream_uri(self) -> Optional[str]:
        try:
            setup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
            resp = self.media.GetStreamUri({"StreamSetup": setup, "ProfileToken": self.profile_token})
            raw_uri = getattr(resp, "Uri", None)
            if not raw_uri:
                return None
            parsed = urlparse(raw_uri)
            if self.username and "@" not in parsed.netloc:
                netloc = f"{self.username}:{self.password}@{parsed.netloc}"
                return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
            return raw_uri
        except Exception:
            return None

    def move_continuous(self, pan_norm: float, tilt_norm: float, zoom_norm: float) -> None:
        px = float(np.clip(pan_norm, -1.0, 1.0))
        py = float(np.clip(tilt_norm, -1.0, 1.0))
        zx = float(np.clip(zoom_norm, -1.0, 1.0))

        if self._supports_joint_zoom:
            try:
                self.ptz.ContinuousMove({
                    "ProfileToken": self.profile_token,
                    "Velocity": {"PanTilt": {"x": px, "y": py}, "Zoom": {"x": zx}},
                })
                return
            except Exception:
                self._supports_joint_zoom = False

        self.ptz.ContinuousMove({
            "ProfileToken": self.profile_token,
            "Velocity": {"PanTilt": {"x": px, "y": py}},
        })
        if abs(zx) > 0.05:
            try:
                self.ptz.ContinuousMove({
                    "ProfileToken": self.profile_token,
                    "Velocity": {"Zoom": {"x": zx}},
                })
            except Exception:
                pass

    def stop(self) -> None:
        try:
            self.ptz.Stop({"ProfileToken": self.profile_token, "PanTilt": True, "Zoom": True})
        except Exception:
            self.ptz.Stop({"ProfileToken": self.profile_token})

    def goto_preset(self, preset_id: int) -> None:
        self.ptz.GotoPreset({"ProfileToken": self.profile_token, "PresetToken": str(preset_id)})


class DahuaCGIPTZDriver(BasePTZDriver):
    """大华 HTTP CGI 物理云台驱动"""

    def __init__(self, ip: str, username: str, password: str, channel: int = 1):
        self.base_url = f"http://{ip}/cgi-bin/ptz.cgi"
        self.auth = HTTPDigestAuth(username, password)
        self.channel = channel
        self._last_code = "Right"
        self._session = requests.Session()

    def move_continuous(self, pan_norm: float, tilt_norm: float, zoom_norm: float) -> None:
        pan_lv = int(round(np.clip(pan_norm, -1.0, 1.0) * 8))
        tilt_lv = int(round(np.clip(tilt_norm, -1.0, 1.0) * 8))
        if pan_lv == 0 and tilt_lv == 0:
            self.stop()
            return

        if pan_lv > 0 and tilt_lv > 0:
            code, a1, a2 = "RightUp", abs(tilt_lv), abs(pan_lv)
        elif pan_lv > 0 and tilt_lv < 0:
            code, a1, a2 = "RightDown", abs(tilt_lv), abs(pan_lv)
        elif pan_lv < 0 and tilt_lv > 0:
            code, a1, a2 = "LeftUp", abs(tilt_lv), abs(pan_lv)
        elif pan_lv < 0 and tilt_lv < 0:
            code, a1, a2 = "LeftDown", abs(tilt_lv), abs(pan_lv)
        elif pan_lv != 0:
            code, a1, a2 = ("Right" if pan_lv > 0 else "Left"), 0, abs(pan_lv)
        else:
            code, a1, a2 = ("Up" if tilt_lv > 0 else "Down"), 0, abs(tilt_lv)

        self._last_code = code
        self._session.get(
            self.base_url,
            params={"action": "start", "channel": self.channel, "code": code, "arg1": a1, "arg2": a2, "arg3": 0},
            auth=self.auth,
            timeout=0.3,
        )

    def stop(self) -> None:
        self._session.get(
            self.base_url,
            params={"action": "stop", "channel": self.channel, "code": self._last_code, "arg1": 0, "arg2": 0, "arg3": 0},
            auth=self.auth,
            timeout=0.3,
        )

    def goto_preset(self, preset_id: int) -> None:
        self._session.get(
            self.base_url,
            params={"action": "start", "channel": self.channel, "code": "GotoPreset", "arg1": 0, "arg2": preset_id, "arg3": 0},
            auth=self.auth,
            timeout=0.3,
        )


class HikvisionISAPIDriver(BasePTZDriver):
    """海康威视 ISAPI 物理云台驱动"""

    def __init__(self, ip: str, username: str, password: str, channel: int = 1):
        self.base_url = f"http://{ip}/ISAPI/PTZCtrl/channels/{channel}/continuous"
        self.preset_url_tpl = f"http://{ip}/ISAPI/PTZCtrl/channels/{channel}/presets/{{preset_id}}/goto"
        self.auth = HTTPDigestAuth(username, password)
        self._session = requests.Session()

    def move_continuous(self, pan_norm: float, tilt_norm: float, zoom_norm: float) -> None:
        pan = int(np.clip(round(pan_norm * 100), -100, 100))
        tilt = int(np.clip(round(tilt_norm * 100), -100, 100))
        zoom = int(np.clip(round(zoom_norm * 100), -100, 100))
        xml_payload = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<PTZData><pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom></PTZData>"
        )
        self._session.put(
            self.base_url,
            data=xml_payload,
            auth=self.auth,
            headers={"Content-Type": "application/xml"},
            timeout=0.3,
        )

    def stop(self) -> None:
        self.move_continuous(0.0, 0.0, 0.0)

    def goto_preset(self, preset_id: int) -> None:
        self._session.put(self.preset_url_tpl.format(preset_id=preset_id), auth=self.auth, timeout=0.3)


class DryRunVerifyPTZDriver(BasePTZDriver):
    """仅用于无硬件连接时的自检与日志输出"""

    def __init__(self, label: str = "DRY-RUN"):
        self.label = label
        self.last_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def move_continuous(self, pan_norm: float, tilt_norm: float, zoom_norm: float) -> None:
        self.last_cmd = (pan_norm, tilt_norm, zoom_norm)

    def stop(self) -> None:
        self.last_cmd = (0.0, 0.0, 0.0)

    def goto_preset(self, preset_id: int) -> None:
        pass


def auto_probe_ptz_driver(
    ip: Optional[str],
    username: str,
    password: str,
    protocol: str = "auto",
    onvif_port: int = 80,
    channel: int = 1,
) -> BasePTZDriver:
    """
    自动探测物理云台控制协议：
    优先轮询 ONVIF 常见端口 (80, 8899, 8080, 5000, 8000，覆盖天云伟业/雄迈公版/宇视/海康/大华)
    -> 海康 ISAPI -> 大华 CGI
    """
    protocol = protocol.lower().strip()
    if not ip:
        return DryRunVerifyPTZDriver()

    if protocol in ("onvif", "auto"):
        seen_ports = []
        # 8899 与 80 是天云伟业等国内公版方案最常用的两个 ONVIF 端口
        for p in (onvif_port, 80, 8899, 8080, 5000, 8000):
            if p not in seen_ports:
                seen_ports.append(p)
        for candidate_port in seen_ports:
            try:
                driver = ONVIFPTZDriver(ip, candidate_port, username, password)
                print(f"[PTZ-PROBE] 成功连接 ONVIF 物理云台 ({ip}:{candidate_port})")
                return driver
            except Exception:
                if protocol == "onvif" and candidate_port == seen_ports[-1]:
                    raise

    if protocol in ("isapi", "auto"):
        try:
            resp = requests.get(
                f"http://{ip}/ISAPI/System/deviceInfo",
                auth=HTTPDigestAuth(username, password),
                timeout=1.2,
            )
            if resp.status_code == 200 and ("Hikvision" in resp.text or "<DeviceInfo" in resp.text):
                print(f"[PTZ-PROBE] 探测到海康 ISAPI 协议支持 ({ip})")
                return HikvisionISAPIDriver(ip, username, password, channel=channel)
        except requests.RequestException:
            if protocol == "isapi":
                raise

    if protocol in ("dahua", "auto"):
        try:
            resp = requests.get(
                f"http://{ip}/cgi-bin/magicBox.cgi?action=getSystemInfo",
                auth=HTTPDigestAuth(username, password),
                timeout=1.2,
            )
            if resp.status_code == 200:
                print(f"[PTZ-PROBE] 探测到大华 CGI 协议支持 ({ip})")
                return DahuaCGIPTZDriver(ip, username, password, channel=channel)
        except requests.RequestException:
            if protocol == "dahua":
                raise

    raise RuntimeError(
        f"无法在 {ip} 上探测到可用的云台控制协议（已尝试 ONVIF 端口 80/8899/8080/5000/8000 及 ISAPI/CGI）。\n"
        f"请先运行 `python3 {Path(__file__).name} --diagnose-ip {ip} --ptz-user {username} --ptz-pass {password}` 排查端口与 ONVIF 开关。"
    )


# =====================================================================
# 4. 非 360° 限位保护异步云台控制器 (防撞死限位 + 自动触发跨镜接力)
# =====================================================================

class LimitAwarePTZController:
    """
    针对“不支持 360° 旋转（带物理机械限位）”球机的闭环控制器：
    1. 行程积分软限位保护 (`max_pan_travel_sec`, `max_tilt_travel_sec`):
       实时积分云台从初始中心位向左右/上下累计转过的行程；
       一旦达到左/右机械限位阈值，立即禁止继续往该方向输出 Pan/Tilt 速度（强制置 0），
       防止电机持续顶死物理限位挡块导致发热、打齿或丢步。
    2. 限位尽头自动触发跨镜接力 (`limit_reached_dir`):
       当云台已经顶到左/右限位，且目标仍在往该侧边缘移动时，立即返回 `"left"` 或 `"right"`，
       通知多相机协同中心唤醒相邻摄像头接管跟踪！
    """

    def __init__(
        self,
        camera_id: str,
        driver: BasePTZDriver,
        deadzone_xy: float = 0.08,
        deadzone_zoom: float = 0.10,
        kp_pan: float = 0.60,
        kd_pan: float = 0.12,
        kv_pan: float = 0.22,
        kp_tilt: float = 0.50,
        kd_tilt: float = 0.10,
        kv_tilt: float = 0.18,
        kp_zoom: float = 0.45,
        max_pan_travel_sec: float = 6.0,   # 从中心转到单侧物理限位的最大等效满速时长（秒）
        max_tilt_travel_sec: float = 3.0,  # 垂直方向单侧最大等效满速时长（秒）
        send_hz: float = 15.0,
    ):
        self.camera_id = camera_id
        self.driver = driver
        self.deadzone_xy = deadzone_xy
        self.deadzone_zoom = deadzone_zoom
        self.kp_pan, self.kd_pan, self.kv_pan = kp_pan, kd_pan, kv_pan
        self.kp_tilt, self.kd_tilt, self.kv_tilt = kp_tilt, kd_tilt, kv_tilt
        self.kp_zoom = kp_zoom
        self.max_pan_travel = max_pan_travel_sec
        self.max_tilt_travel = max_tilt_travel_sec
        self.send_interval = 1.0 / max(1.0, send_hz)

        self._lock = threading.Lock()
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self.pan_pos_est = 0.0
        self.tilt_pos_est = 0.0
        self._last_update_ts = time.monotonic()

        self._desired_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_sent_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._pending_preset: Optional[int] = None
        self._is_tracking_active = False

        self._running = True
        self._thread = threading.Thread(
            target=self._tx_loop, name=f"PTZ-Loop-{camera_id}", daemon=True
        )
        self._thread.start()

    def update_control_nonblocking(
        self,
        norm_err_x: float,
        norm_err_y: float,
        zoom_err: float,
        norm_vel_x: float = 0.0,
        norm_vel_y: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], Optional[str]]:
        now = time.monotonic()
        dt = min(0.2, max(0.0, now - self._last_update_ts))
        self._last_update_ts = now

        d_err_x = norm_err_x - self._prev_err_x
        d_err_y = norm_err_y - self._prev_err_y
        self._prev_err_x = norm_err_x
        self._prev_err_y = norm_err_y

        pan_v = 0.0
        tilt_v = 0.0
        zoom_v = 0.0
        limit_reached_dir: Optional[str] = None

        if abs(norm_err_x) > self.deadzone_xy or abs(norm_vel_x) > 0.15:
            pan_v = float(
                np.clip(
                    self.kp_pan * norm_err_x + self.kd_pan * d_err_x + self.kv_pan * norm_vel_x,
                    -1.0,
                    1.0,
                )
            )

        if abs(norm_err_y) > self.deadzone_xy or abs(norm_vel_y) > 0.15:
            tilt_v = float(
                np.clip(
                    -(self.kp_tilt * norm_err_y + self.kd_tilt * d_err_y + self.kv_tilt * norm_vel_y),
                    -1.0,
                    1.0,
                )
            )

        if abs(zoom_err) > self.deadzone_zoom:
            zoom_v = float(np.clip(self.kp_zoom * zoom_err, -1.0, 1.0))

        with self._lock:
            # 非 360° 球机水平限位保护：若已到达最右/最左限位，切断同向 Pan 电机指令并强制拉广角
            if pan_v > 0 and self.pan_pos_est >= self.max_pan_travel:
                pan_v = 0.0
                zoom_v = min(zoom_v, -0.5)
                limit_reached_dir = "right"
            elif pan_v < 0 and self.pan_pos_est <= -self.max_pan_travel:
                pan_v = 0.0
                zoom_v = min(zoom_v, -0.5)
                limit_reached_dir = "left"
            else:
                self.pan_pos_est = float(
                    np.clip(self.pan_pos_est + pan_v * dt, -self.max_pan_travel, self.max_pan_travel)
                )

            if (tilt_v > 0 and self.tilt_pos_est >= self.max_tilt_travel) or (
                tilt_v < 0 and self.tilt_pos_est <= -self.max_tilt_travel
            ):
                tilt_v = 0.0
            else:
                self.tilt_pos_est = float(
                    np.clip(self.tilt_pos_est + tilt_v * dt, -self.max_tilt_travel, self.max_tilt_travel)
                )

            cmd = (round(pan_v, 3), round(tilt_v, 3), round(zoom_v, 3))
            self._desired_cmd = cmd
            self._is_tracking_active = True

        return cmd, limit_reached_dir

    def stop_nonblocking(self) -> None:
        with self._lock:
            self._desired_cmd = (0.0, 0.0, 0.0)
            self._is_tracking_active = False
            self._prev_err_x = 0.0
            self._prev_err_y = 0.0

    def trigger_handoff_preset_if_idle(self, preset_id: int) -> bool:
        with self._lock:
            if self._is_tracking_active:
                return False
            self._pending_preset = preset_id
            self.pan_pos_est = 0.0
            self.tilt_pos_est = 0.0
            return True

    def shutdown(self) -> None:
        self.stop_nonblocking()
        time.sleep(self.send_interval * 1.2)
        self._running = False

    def _tx_loop(self) -> None:
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

            try:
                if preset_to_call is not None:
                    self.driver.goto_preset(preset_to_call)
                elif cmd_to_send is not None:
                    if cmd_to_send == (0.0, 0.0, 0.0):
                        self.driver.stop()
                    else:
                        self.driver.move_continuous(*cmd_to_send)
            except Exception as exc:
                print(f"[{self.camera_id} PTZ-ERROR] 云台指令执行失败: {exc}")

            elapsed = time.monotonic() - t0
            time.sleep(max(0.005, self.send_interval - elapsed))


# =====================================================================
# 5. 真实模型加载器 (YOLO11-Pose + InsightFace + Torchreid/ResNet50)
# =====================================================================

class RealVisionModelBundle:
    def __init__(
        self,
        pose_weights: str = "yolo11n-pose.pt",
        reid_weights: str = "osnet_x1_0",
        device: str = "cuda:0",
        require_face_model: bool = True,
    ):
        self.device = device
        self.pose_weights = pose_weights
        self.reid_weights = reid_weights
        self.face_app = None
        self.reid_extractor = None

        if require_face_model:
            from insightface.app import FaceAnalysis

            ctx_id = 0 if "cuda" in device else -1
            self.face_app = FaceAnalysis(name="buffalo_l")
            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        try:
            from torchreid.utils import FeatureExtractor

            self.reid_extractor = FeatureExtractor(
                model_name=reid_weights,
                device=device if "cuda" in device else "cpu",
            )
        except ImportError:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone.fc = torch.nn.Identity()
            backbone.eval().to(device if torch.cuda.is_available() and "cuda" in device else "cpu")
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._torch_dev = next(backbone.parameters()).device

            def _resnet_reid(crops: List[np.ndarray]) -> List[np.ndarray]:
                tensors = torch.stack([transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]).to(self._torch_dev)
                with torch.no_grad():
                    feats = backbone(tensors).cpu().numpy()
                return [f / (np.linalg.norm(f) + 1e-6) for f in feats]

            self.reid_extractor = _resnet_reid

    def create_camera_pose_tracker(self):
        from ultralytics import YOLO

        return YOLO(self.pose_weights)

    def extract_face_embeddings(self, bgr_img: np.ndarray) -> List[Dict]:
        if self.face_app is None:
            return []
        faces = self.face_app.get(bgr_img)
        return [
            {"bbox": f.bbox, "embedding": f.normed_embedding}
            for f in faces
            if getattr(f, "normed_embedding", None) is not None
        ]

    def extract_body_reid(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        if self.reid_extractor is None or bgr_crop is None or bgr_crop.size == 0:
            return None
        feats = self.reid_extractor([bgr_crop])
        vec = feats[0].cpu().numpy() if hasattr(feats[0], "cpu") else np.asarray(feats[0])
        return vec / (np.linalg.norm(vec) + 1e-6)


# =====================================================================
# 6. 全局白名单特征库与多摄像头拓扑中心
# =====================================================================

@dataclass
class TargetPersonProfile:
    person_id: str
    name: str
    face_embedding: np.ndarray
    reid_gallery: List[np.ndarray] = field(default_factory=list)
    max_gallery_size: int = 16
    last_seen_camera: Optional[str] = None
    last_seen_timestamp: float = 0.0

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

    def load_whitelist_from_directory(self, whitelist_dir: str, model_bundle: RealVisionModelBundle) -> int:
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(glob.glob(os.path.join(whitelist_dir, ext)))

        loaded_count = 0
        for path in sorted(img_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            stem = Path(path).stem
            faces = model_bundle.extract_face_embeddings(img)
            if not faces:
                print(f"[WHITELIST-WARN] 照片 {path} 未检测到清晰人脸，跳过注册")
                continue
            face_emb = faces[0]["embedding"]
            self.register_whitelist_person(person_id=stem, name=stem, face_embedding=face_emb)
            loaded_count += 1
            print(f"[WHITELIST-OK] 已注册目标人物: {stem} ({path})")
        return loaded_count

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
                if (raw_score + bonus) > best_score:
                    best_score = raw_score + bonus
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
            if body_embedding is not None:
                profile.update_reid_feature(body_embedding)
            if exit_direction is not None:
                for neighbor_cam, preset_id in self.topology_presets.get(camera_id, {}).items():
                    pre_aim_tasks.append((neighbor_cam, preset_id))
        return pre_aim_tasks


# =====================================================================
# 7. 侧身躯干锚点 + 150ms 速度前馈 + 策略 A 多人视野包络解算
# =====================================================================

def compute_torso_anchor(bbox: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """优先使用 COCO 17 关键点中的双肩(5,6)与双髋(11,12)计算躯干质心，侧身/转身时锁定点不漂移"""
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

    pred_bboxes, pred_anchors, vel_list = [], [], []
    for bbox, (ax, ay), (vx, vy) in locked_targets:
        dx, dy = vx * lookahead_sec, vy * lookahead_sec
        pred_bboxes.append([bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy])
        pred_anchors.append([ax + dx, ay + dy])
        vel_list.append([vx / (frame_w * 0.5), vy / (frame_h * 0.5)])

    bboxes_arr = np.array(pred_bboxes)
    anchors_arr = np.array(pred_anchors)
    vels_arr = np.array(vel_list)

    env_x1, env_y1 = float(np.min(bboxes_arr[:, 0])), float(np.min(bboxes_arr[:, 1]))
    env_x2, env_y2 = float(np.max(bboxes_arr[:, 2])), float(np.max(bboxes_arr[:, 3]))

    if len(locked_targets) == 1:
        aim_x, aim_y = float(anchors_arr[0, 0]), float(anchors_arr[0, 1])
    else:
        env_cx, env_cy = (env_x1 + env_x2) * 0.5, (env_y1 + env_y2) * 0.5
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
    zoom_err = -0.65 if near_edge else (target_fill_ratio - span_ratio)

    return norm_err_x, norm_err_y, zoom_err, norm_vel_x, norm_vel_y


# =====================================================================
# 8. 慢轨异步身份池 & 快轨单相机节点 (有界整帧全景 + 限位触发跨镜接力)
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
        model_bundle: Optional[RealVisionModelBundle],
        max_queue_size: int = 32,
    ):
        self.hub = topology_hub
        self.models = model_bundle
        self.task_queue: "queue.Queue[AsyncIdentityTask]" = queue.Queue(maxsize=max_queue_size)
        self.camera_workers: Dict[str, "FastCameraTrackerNode"] = {}
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="SlowTrack-Pool", daemon=True)
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

    def _loop(self) -> None:
        while self._running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            node = self.camera_workers.get(task.camera_id)
            if node is None or self.models is None:
                continue

            if task.is_gallery_update and task.known_person_id:
                body_emb = self.models.extract_body_reid(task.person_crop)
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

            faces = self.models.extract_face_embeddings(task.person_crop)
            matched_pid = None
            for f in faces:
                matched_pid = self.hub.match_face(f["embedding"])
                if matched_pid:
                    break

            body_emb = None
            if matched_pid is None:
                body_emb = self.models.extract_body_reid(task.person_crop)
                if body_emb is not None:
                    matched_pid = self.hub.match_reid_with_topology(body_emb, task.camera_id)

            if matched_pid is not None:
                if body_emb is None:
                    body_emb = self.models.extract_body_reid(task.person_crop)
                node.bind_track_to_person(task.track_id, matched_pid)
                self.hub.update_person_sighting(matched_pid, task.camera_id, body_embedding=body_emb)


class FastCameraTrackerNode:
    def __init__(
        self,
        camera_id: str,
        ptz_controller: LimitAwarePTZController,
        slow_pool: SlowTrackIdentityWorkerPool,
        pose_tracker_model,
        enable_undistort: bool = False,
    ):
        self.camera_id = camera_id
        self.ptz = ptz_controller
        self.slow_pool = slow_pool
        self.pose_model = pose_tracker_model
        self.pano_processor = BoundedPanoramaProcessor(enable_undistort=enable_undistort)

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

    def process_live_frame(
        self, raw_frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]]]:
        self.frame_idx += 1
        now = time.monotonic()

        active_frame = self.pano_processor.preprocess_full_frame(raw_frame)
        h, w = active_frame.shape[:2]

        tracks = self._run_yolo_pose_bytetrack(active_frame)
        locked_targets: Dict[str, Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]] = {}
        person_crops: Dict[str, Tuple[int, np.ndarray, Optional[str]]] = {}

        for trk in tracks:
            track_id: int = trk["track_id"]
            bbox: np.ndarray = trk["bbox"]
            keypoints: Optional[np.ndarray] = trk.get("keypoints")

            ax, ay = compute_torso_anchor(bbox, keypoints)
            vx, vy = self._update_velocity(track_id, ax, ay, now)
            person_id = self.get_bound_person(track_id)

            if person_id is None:
                if self.frame_idx % 3 == 1:
                    crop = self._crop_person(active_frame, bbox)
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
                exit_dir = "left" if (ax < w * 0.12 and vx < -30.0) else ("right" if (ax > w * 0.88 and vx > 30.0) else None)
                if (self.frame_idx % 12 == 0) or (exit_dir is not None):
                    crop = self._crop_person(active_frame, bbox)
                    if crop is not None:
                        person_crops[person_id] = (track_id, crop, exit_dir)

        if locked_targets:
            err_x, err_y, zoom_err, vel_x, vel_y = compute_strategy_a_with_velocity_feedforward(
                list(locked_targets.values()), frame_w=w, frame_h=h
            )
            _, hit_limit_dir = self.ptz.update_control_nonblocking(err_x, err_y, zoom_err, vel_x, vel_y)

            for pid, (tid, crop, exit_dir) in person_crops.items():
                effective_exit = hit_limit_dir or exit_dir
                self.slow_pool.submit_task_nonblocking(
                    AsyncIdentityTask(
                        camera_id=self.camera_id,
                        track_id=tid,
                        person_crop=crop,
                        is_gallery_update=True,
                        known_person_id=pid,
                        exit_direction=effective_exit,
                    )
                )
        else:
            self.ptz.stop_nonblocking()

        return active_frame, locked_targets

    def _run_yolo_pose_bytetrack(self, frame: np.ndarray) -> List[Dict]:
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

    def _update_velocity(self, track_id: int, ax: float, ay: float, now: float) -> Tuple[float, float]:
        if track_id not in self._track_kinematics:
            self._track_kinematics[track_id] = (ax, ay, 0.0, 0.0, now)
            return 0.0, 0.0
        prev_x, prev_y, prev_vx, prev_vy, prev_t = self._track_kinematics[track_id]
        dt = max(1e-3, now - prev_t)
        vx = 0.6 * prev_vx + 0.4 * ((ax - prev_x) / dt)
        vy = 0.6 * prev_vy + 0.4 * ((ay - prev_y) / dt)
        self._track_kinematics[track_id] = (ax, ay, vx, vy, now)
        return vx, vy

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox[:4]]
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return None
        return frame[y1:y2, x1:x2].copy()


# =====================================================================
# 9. CLI 入口与非 360° 限位保护自检
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多品牌通用 (适配天云伟业/ONVIF/大华/海康) & 有界整帧全景 + 非360°限位保护云台跟踪系统"
    )
    parser.add_argument(
        "--diagnose-ip",
        type=str,
        default="",
        help="一键诊断指定 IP 摄像头（如天云伟业球机）开放的端口、ONVIF 支持情况与 RTSP 地址",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="多路视频源（RTSP URL 或本地视频路径），格式: CAM_01=rtsp://...",
    )
    parser.add_argument(
        "--ptz-ips",
        nargs="*",
        default=[],
        help="各摄像头对应的云台控制 IP，格式: CAM_01=192.168.1.10 CAM_02=192.168.1.11",
    )
    parser.add_argument("--ptz-user", type=str, default="admin", help="相机登录用户名")
    parser.add_argument("--ptz-pass", type=str, default="123456", help="相机登录密码")
    parser.add_argument("--onvif-port", type=int, default=80, help="首选 ONVIF 端口（自动轮询 80, 8899, 8080, 5000, 8000）")
    parser.add_argument(
        "--ptz-protocol",
        type=str,
        choices=["auto", "onvif", "isapi", "dahua"],
        default="auto",
        help="云台控制协议（默认 auto 自动探测 ONVIF -> 海康 ISAPI -> 大华 CGI）",
    )
    parser.add_argument(
        "--max-pan-travel-sec",
        type=float,
        default=6.0,
        help="非360°球机从中心转到左/右物理限位的最大等效满速时长（秒），到达后自动停转防撞限位并触发跨镜接力",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="对有界整帧全景/半球广角画面开启径向畸变矫正（不作360°环绕拼接）",
    )
    parser.add_argument(
        "--whitelist-dir",
        type=str,
        default="./whitelist_faces",
        help="预录入白名单人物照片文件夹路径（如 ./whitelist_faces/张三.jpg）",
    )
    parser.add_argument("--pose-weights", type=str, default="yolo11n-pose.pt", help="YOLO11-Pose 权重路径")
    parser.add_argument("--reid-weights", type=str, default="osnet_x1_0", help="Torchreid 模型名称或权重路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备 (cuda:0 或 cpu)")
    parser.add_argument("--self-test", action="store_true", help="运行有界整帧全景矫正与非360°限位保护自检")
    return parser.parse_args()


def run_self_test() -> None:
    print("=== [自检 1] 有界整帧全景（非360°环绕）广角畸变矫正验证 ===")
    raw_pano = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(raw_pano, (100, 200), (1820, 880), (0, 200, 255), 4)
    proc = BoundedPanoramaProcessor(enable_undistort=True)
    t0 = time.perf_counter()
    rectified = proc.preprocess_full_frame(raw_pano)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"有界整帧全景矫正输出尺寸: {rectified.shape}, 首次建表+remap耗时: {dt_ms:.2f} ms")

    print("=== [自检 2] 非 360° 球机物理限位保护（防堵转 + 自动拉广角 + 触发跨镜接力）验证 ===")
    dry_driver = DryRunVerifyPTZDriver()
    ctrl = LimitAwarePTZController("CAM_01", dry_driver, max_pan_travel_sec=0.20)
    for step in range(5):
        cmd, hit_limit = ctrl.update_control_nonblocking(
            norm_err_x=0.80, norm_err_y=0.0, zoom_err=0.0, norm_vel_x=0.50
        )
        print(
            f"  Step {step + 1}: 累计行程估计={ctrl.pan_pos_est:.3f}s -> "
            f"输出指令(pan,tilt,zoom)={cmd}, 到达物理限位触发跨镜接力={hit_limit}"
        )
        time.sleep(0.12)
    ctrl.shutdown()


def main() -> None:
    args = parse_args()
    if args.diagnose_ip:
        diagnose_camera_compatibility(args.diagnose_ip.strip(), args.ptz_user, args.ptz_pass)
        return

    if args.self_test or (not args.sources and not args.ptz_ips):
        if not args.sources and not args.ptz_ips and not args.self_test:
            print(
                "提示：未指定 `--sources` / `--ptz-ips` 或 `--diagnose-ip`，执行内置有界全景与非360°限位保护自检。\n"
                "若要测试天云伟业球机能否直接控制，请运行:\n"
                f"  python3 {Path(__file__).name} --diagnose-ip <球机IP> --ptz-user <用户名> --ptz-pass <密码>\n"
            )
        run_self_test()
        return

    models = RealVisionModelBundle(
        pose_weights=args.pose_weights,
        reid_weights=args.reid_weights,
        device=args.device,
    )
    hub = CrossCameraTopologyHub()
    if os.path.isdir(args.whitelist_dir):
        hub.load_whitelist_from_directory(args.whitelist_dir, models)
    else:
        print(f"[WARN] 白名单目录 {args.whitelist_dir} 不存在，请创建并放入目标人物正面照。")

    slow_pool = SlowTrackIdentityWorkerPool(topology_hub=hub, model_bundle=models)

    ip_map: Dict[str, str] = {}
    for item in args.ptz_ips:
        if "=" in item:
            cid, ip_val = item.split("=", 1)
            ip_map[cid.strip()] = ip_val.strip()

    source_map: Dict[str, str] = {}
    for idx, src_item in enumerate(args.sources):
        if "=" in src_item:
            cid, url = src_item.split("=", 1)
        else:
            cid, url = f"CAM_{idx + 1:02d}", src_item
        source_map[cid.strip()] = url.strip()

    all_cam_ids = sorted(set(list(source_map.keys()) + list(ip_map.keys())))
    for i in range(len(all_cam_ids) - 1):
        hub.register_camera_link(all_cam_ids[i], all_cam_ids[i + 1], entry_preset_id=1)

    for cid in all_cam_ids:
        driver = auto_probe_ptz_driver(
            ip=ip_map.get(cid),
            username=args.ptz_user,
            password=args.ptz_pass,
            protocol=args.ptz_protocol,
            onvif_port=args.onvif_port,
        )
        stream_url = source_map.get(cid)
        if not stream_url and isinstance(driver, ONVIFPTZDriver):
            stream_url = driver.get_rtsp_stream_uri()
            if stream_url:
                print(f"[{cid} ONVIF-RTSP] 已通过 ONVIF 自动获取到 RTSP 地址: {stream_url}")
        if not stream_url:
            raise RuntimeError(f"摄像头 {cid} 未提供 `--sources` RTSP 地址，且无法通过 ONVIF 自动获取流地址")

        ptz_ctrl = LimitAwarePTZController(
            cid,
            driver,
            max_pan_travel_sec=args.max_pan_travel_sec,
        )
        pose_tracker = models.create_camera_pose_tracker()
        node = FastCameraTrackerNode(
            camera_id=cid,
            ptz_controller=ptz_ctrl,
            slow_pool=slow_pool,
            pose_tracker_model=pose_tracker,
            enable_undistort=args.undistort,
        )

        def _worker(cam_node: FastCameraTrackerNode, url_to_open: str) -> None:
            cap = cv2.VideoCapture(int(url_to_open) if url_to_open.isdigit() else url_to_open)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while cap.isOpened():
                if not cap.grab():
                    time.sleep(0.005)
                    continue
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue
                cam_node.process_live_frame(frame)
            cap.release()

        threading.Thread(target=_worker, args=(node, stream_url), name=f"Worker-{cid}", daemon=True).start()

    print(f"[RUNNING] 已启动 {len(all_cam_ids)} 路摄像头实时锁定跟随（非360°限位保护已启用），按 Ctrl+C 退出...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        slow_pool.shutdown()


if __name__ == "__main__":
    main()
