#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import numpy as np
import cv2
import torch
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

# --- Video Depth Anything (stream) ---
# 레포를 워크스페이스에 포함하거나 PYTHONPATH를 잡아주세요.
# 예) export PYTHONPATH=$PYTHONPATH:/path/to/Video-Depth-Anything
from video_depth_anything.video_depth_stream import VideoDepthAnything  # :contentReference[oaicite:3]{index=3}


def depth_to_points(depth_m, fx, fy, cx, cy, stride=4, rgb=None):
    """
    Auto Projection: 픽셀 깊이 z를 카메라 내참수로 3D (X,Y,Z)로 역투영.
    depth_m: (H, W) [meters]
    rgb: (H, W, 3) uint8 or None (컬러 샘플링)
    """
    H, W = depth_m.shape[:2]
    ys = np.arange(0, H, stride, dtype=np.float32)
    xs = np.arange(0, W, stride, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)  # u=x(col), v=y(row)
    z = depth_m[v.astype(int), u.astype(int)]
    valid = np.isfinite(z) & (z > 0.0)

    u = u[valid]; v = v[valid]; z = z[valid]
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    pts = np.stack([X, Y, z], axis=1)  # (N,3)

    colors = None
    if rgb is not None:
        colors = rgb[v.astype(int), u.astype(int)] / 255.0  # (N,3) in [0,1]
    return pts, colors


def o3d_mesh_to_marker(mesh: o3d.geometry.TriangleMesh, frame_id: str, rgba=(0.2, 0.7, 1.0, 0.8)):
    """
    Open3D TriangleMesh -> RViz Marker(TRIANGLE_LIST)
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "vda_bpa_mesh"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
    marker.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    # TRIANGLE_LIST는 3개 점씩 push
    pts = []
    for a, b, c in tris:
        for idx in (a, b, c):
            p = Point()
            p.x, p.y, p.z = float(verts[idx, 0]), float(verts[idx, 1]), float(verts[idx, 2])
            pts.append(p)
    marker.points = pts
    return marker


class VideoDepthAnythingStreamNode(Node):
    def __init__(self):
        super().__init__("video_depth_anything_stream_node")
        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("encoder", "vits")      # Small
        self.declare_parameter("metric", True)         # Metric-Video-Depth-Anything
        self.declare_parameter("input_size", 518)
        self.declare_parameter("max_res", 1280)        # 긴 변 제한
        self.declare_parameter("fp32", False)          # CUDA 없으면 자동으로 True 강제
        self.declare_parameter("checkpoint_path", "./checkpoints/metric_video_depth_anything_vits.pth")
        self.declare_parameter("process_hz", 1.0)      # 1 fps
        self.declare_parameter("pcd_stride", 4)        # 포인트 샘플링간격
        self.declare_parameter("bpa_factor_list", [1.5, 2.0, 2.5])  # 평균 NN 거리 배수
        self.declare_parameter("max_triangles", 200_000)  # RViz 과부하 방지용 하드캡
        self.declare_parameter("mesh_color_rgba", [0.2, 0.7, 1.0, 0.8])

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.encoder = self.get_parameter("encoder").get_parameter_value().string_value
        self.metric = self.get_parameter("metric").get_parameter_value().bool_value
        self.input_size = int(self.get_parameter("input_size").get_parameter_value().integer_value)
        self.max_res = int(self.get_parameter("max_res").get_parameter_value().integer_value)
        self.fp32 = self.get_parameter("fp32").get_parameter_value().bool_value
        self.checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value
        self.process_hz = float(self.get_parameter("process_hz").get_parameter_value().double_value)
        self.pcd_stride = int(self.get_parameter("pcd_stride").get_parameter_value().integer_value)
        self.bpa_factor_list = list(self.get_parameter("bpa_factor_list").get_parameter_value().double_array_value)
        self.max_triangles = int(self.get_parameter("max_triangles").get_parameter_value().integer_value)
        self.mesh_color_rgba = list(self.get_parameter("mesh_color_rgba").get_parameter_value().double_array_value)

        # Device & dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            self.fp32 = True  # CPU에서는 fp32 강제
        self.get_logger().info(f"Device: {self.device} | fp32={self.fp32}")

        # ---- Load VDA Stream Model (Small / Metric) ----
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        if self.encoder not in model_configs:
            raise ValueError(f"Unknown encoder: {self.encoder}")

        # metric=True이면 metric_video_depth_anything_* 체크포인트를 사용 (레포 규칙 참조) :contentReference[oaicite:4]{index=4}
        if not os.path.isfile(self.checkpoint_path):
            self.get_logger().warn(f"Checkpoint not found at {self.checkpoint_path}. Make sure it exists.")

        self.vda = VideoDepthAnything(**model_configs[self.encoder])
        state = torch.load(self.checkpoint_path, map_location='cpu')
        self.vda.load_state_dict(state, strict=True)
        self.vda = self.vda.to(self.device).eval()

        # ---- ROS I/O ----
        self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.pub_mesh = self.create_publisher(Marker, "vda/mesh_marker", 1)
        self.pub_cloud = self.create_publisher(PointCloud2, "vda/pointcloud", 1)

        # 상태
        self.lock = threading.Lock()
        self.latest_img_msg = None
        self.latest_bgr = None
        self.K = None
        self.frame_id = "camera_color_optical_frame"

        # 타이머: 1 Hz 처리
        self.timer = self.create_timer(1.0 / self.process_hz, self.process_tick)
        self.get_logger().info("VideoDepthAnything stream node is up.")

    # --- Subscribers ---
    def on_camera_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.frame_id = msg.header.frame_id or self.frame_id

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return
        with self.lock:
            self.latest_img_msg = msg
            self.latest_bgr = bgr

    def _build_pointcloud_msg(self, pts, colors, header):
        if pts.size == 0:
            return None
        points = np.asarray(pts, dtype=np.float32)

        if colors is None:
            cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
            cloud_msg.is_dense = False
            return cloud_msg

        colors = np.clip(np.asarray(colors), 0.0, 1.0)
        colors_uint8 = (colors * 255.0).astype(np.uint8)
        rgb_uint32 = (
            (colors_uint8[:, 0].astype(np.uint32) << 16)
            | (colors_uint8[:, 1].astype(np.uint32) << 8)
            | colors_uint8[:, 2].astype(np.uint32)
        )
        rgb_float32 = rgb_uint32.view(np.float32)
        points_rgba = np.hstack((points, rgb_float32.reshape(-1, 1)))

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, points_rgba.tolist())
        cloud_msg.is_dense = False
        return cloud_msg

    # --- Main processing (1 Hz) ---
    def process_tick(self):
        with self.lock:
            if self.latest_bgr is None or self.K is None or self.latest_img_msg is None:
                return
            bgr = self.latest_bgr.copy()
            header = self.latest_img_msg.header

        H, W = bgr.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # 1) 전처리: 최대해상도 제한 및 RGB 변환
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.max_res > 0 and max(H, W) > self.max_res:
            scale = self.max_res / float(max(H, W))
            new_hw = (int(W * scale + 0.5), int(H * scale + 0.5))  # (W,H)
            rgb_small = cv2.resize(rgb, new_hw, interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb

        # 2) 스트리밍 추론 (연속 프레임 호출 시 내부 상태 유지)  :contentReference[oaicite:5]{index=5}
        t0 = time.time()
        depth_small = self.vda.infer_video_depth_one(
            rgb_small,
            input_size=int(self.input_size),
            device=self.device,
            fp32=bool(self.fp32),
        )  # numpy (h,w), metric면 [m] 추정
        infer_ms = (time.time() - t0) * 1000.0

        # 3) 원본 크기로 리사이즈
        depth_m = depth_small
        if depth_small.shape[:2] != (H, W):
            depth_m = cv2.resize(depth_small, (W, H), interpolation=cv2.INTER_LINEAR)

        # 4) Auto Projection (역투영 -> 포인트클라우드)
        pts, colors = depth_to_points(depth_m, fx, fy, cx, cy, stride=self.pcd_stride, rgb=rgb)

        cloud_header = Header()
        cloud_header.stamp = header.stamp
        cloud_header.frame_id = self.frame_id

        cloud_msg = self._build_pointcloud_msg(pts, colors, cloud_header)
        if cloud_msg is not None:
            self.pub_cloud.publish(cloud_msg)

        # 5) Open3D 포인트클라우드 & BPA 메시
        if pts.shape[0] < 100:  # 너무 적으면 스킵
            self.get_logger().warn("Too few valid 3D points; skip BPA.")
            return
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 노멀 추정
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        # 평균 최근접거리 기반 반경 리스트
        dists = pcd.compute_nearest_neighbor_distance()
        mean_nn = float(np.mean(dists)) if len(dists) else 0.01
        radii = o3d.utility.DoubleVector([mean_nn * f for f in self.bpa_factor_list])

        # BPA 메쉬 재구성  :contentReference[oaicite:6]{index=6}
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

        # 과부하 방지: 삼각형 수 제한 (옵션)
        tris = np.asarray(mesh.triangles)
        if tris.shape[0] > self.max_triangles:
            self.get_logger().warn(f"Mesh triangles {tris.shape[0]} > cap {self.max_triangles}; downsampling.")
            step = int(np.ceil(tris.shape[0] / self.max_triangles))
            mesh.triangles = o3d.utility.Vector3iVector(tris[::step])

        # 6) RViz Marker로 퍼블리시
        marker = o3d_mesh_to_marker(mesh, frame_id=self.frame_id, rgba=tuple(self.mesh_color_rgba))
        marker.header.stamp = header.stamp
        self.pub_mesh.publish(marker)

        self.get_logger().info(f"Processed frame @ {self.process_hz:.1f}Hz | infer {infer_ms:.1f} ms | "
                               f"pts {pts.shape[0]} | tris {len(np.asarray(mesh.triangles))}")

def main():
    rclpy.init()
    node = VideoDepthAnythingStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
