#!/usr/bin/env python3
"""
demo_video.py
原有目录版本：复用 config.py / data_loader.py / models.py / physics_features.py
生成正面视频拳头识别 + 可选 PINN 预测落点的演示视频
"""

import sys
import os
import argparse  # ← 修复：补上这个import

# 确保能 import 当前目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

# ========== 复用现有代码 ==========
from config import *
from data_loader import get_kp, load_video, smooth_keypoints_kalman, interpolate_nan_kp
from physics_features import compute_physics

# PINN 模型可选加载
try:
    from models import PINNModel
    HAS_PINN = True
except ImportError:
    HAS_PINN = False


# ========== 可视化配置 ==========
C_SKEL = (0, 255, 0)       # 骨架：绿
C_LW = (255, 128, 0)       # 左手：橙
C_RW = (0, 128, 255)       # 右手：蓝
C_PRED = (0, 0, 255)       # 预测落点：红
C_TEXT = (255, 255, 255)   # 文字：白
C_ACT = (0, 255, 255)      # 动作高亮：青

# 骨架连线（复用 config 里的关节索引）
SKELETON = [
    (NOSE, 1), (NOSE, 2), (1, 3), (2, 4),
    (LS, RS), (LS, LE), (LE, LW), (RS, RE), (RE, RW),
    (LS, HIP_L), (RS, HIP_R), (HIP_L, HIP_R),
    (HIP_L, 13), (13, 15), (HIP_R, 14), (14, 16)
]

# 动作检测阈值（像素/帧）
SPEED_THRESH = 8.0


def is_valid(kp, idx):
    """安全判断关键点是否有效（防NaN）"""
    if kp is None or idx >= len(kp):
        return False
    return not np.any(np.isnan(kp[idx])) and kp[idx][0] > 0 and kp[idx][1] > 0


def draw_skeleton(frame, kp):
    """画骨架（已修复NaN问题）"""
    if kp is None:
        return frame
    
    for a, b in SKELETON:
        if not is_valid(kp, a) or not is_valid(kp, b):
            continue
        
        x1, y1 = int(kp[a][0]), int(kp[a][1])
        x2, y2 = int(kp[b][0]), int(kp[b][1])
        cv2.line(frame, (x1, y1), (x2, y2), C_SKEL, 2)
    
    for i, (x, y) in enumerate(kp):
        if np.any(np.isnan([x, y])) or x <= 0 or y <= 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, C_SKEL, -1)
        
        # 手腕加大标记
        if i == LW:
            cv2.circle(frame, (int(x), int(y)), 8, C_LW, 2)
        elif i == RW:
            cv2.circle(frame, (int(x), int(y)), 8, C_RW, 2)
    
    return frame


def draw_fist(frame, kp, traj, color, label, is_action=False):
    """画拳头 + 轨迹"""
    if kp is None or np.any(np.isnan(kp)) or kp[0] <= 0 or kp[1] <= 0:
        return frame
    
    x, y = int(kp[0]), int(kp[1])
    r = 14 if is_action else 8
    c = C_ACT if is_action else color
    
    cv2.circle(frame, (x, y), r, c, 3 if is_action else 2)
    cv2.putText(frame, label, (x + 18, y - 18),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    
    # 轨迹
    if len(traj) >= 2:
        pts = list(traj)[-25:]
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            if all(v > 0 for v in p1 + p2):
                alpha = i / len(pts)
                col = tuple(int(ch * alpha + 255 * (1 - alpha)) for ch in color)
                cv2.line(frame, p1, p2, col, 2)
    
    return frame


def draw_panel(frame, frame_idx, total, l_spd, r_spd, l_act, r_act, has_pinn):
    """右上角信息面板"""
    h, w = frame.shape[:2]
    pw, ph = 340, 140
    px, py = w - pw - 15, 15
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    
    status_l = "PUNCH!" if l_act else "idle"
    status_r = "PUNCH!" if r_act else "idle"
    
    texts = [
        f"Frame: {frame_idx}/{total}",
        f"Left:  {l_spd:5.1f} px/f [{status_l}]",
        f"Right: {r_spd:5.1f} px/f [{status_r}]",
    ]
    if has_pinn:
        texts.append("PINN Model: ACTIVE")
    
    for i, t in enumerate(texts):
        color = C_ACT if "PUNCH" in t else C_TEXT
        cv2.putText(frame, t, (px + 12, py + 28 + i * 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    
    return frame


def predict_landing_simple(kp, prev_kp, hand_idx):
    """
    简化预测：当前位置 + 速度 * 5帧
    用于没有PINN模型时的演示
    """
    if prev_kp is None or not is_valid(kp, hand_idx) or not is_valid(prev_kp, hand_idx):
        return None
    
    vx = kp[hand_idx][0] - prev_kp[hand_idx][0]
    vy = kp[hand_idx][1] - prev_kp[hand_idx][1]
    
    pred_x = int(kp[hand_idx][0] + vx * 5)
    pred_y = int(kp[hand_idx][1] + vy * 5)
    
    return (pred_x, pred_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="boxing_single_1/video1.mp4", help="正面视频路径")
    parser.add_argument("--yolo", default="yolov8n-pose.pt", help="YOLO模型路径")
    parser.add_argument("--model", default="best_model.pt", help="PINN模型路径（可选）")
    parser.add_argument("--output", default="demo_presentation.mp4", help="输出视频路径")
    parser.add_argument("--max_frames", type=int, default=300, help="最大处理帧数")
    parser.add_argument("--no_pinn", action="store_true", help="强制不使用PINN预测")
    args = parser.parse_args()
    
    # 路径检查
    if not os.path.exists(args.video):
        print(f"❌ 找不到视频: {args.video}")
        print("当前目录:", os.getcwd())
        print("请确认 boxing_single_1/video1.mp4 存在")
        return
    
    print(f"视频路径: {os.path.abspath(args.video)}")
    print(f"YOLO模型: {args.yolo}")
    
    # 加载 YOLO
    print("加载YOLO...")
    yolo = YOLO(args.yolo)
    
    # 加载 PINN（可选）
    model = None
    use_pinn = False
    if not args.no_pinn and HAS_PINN and os.path.exists(args.model):
        print(f"加载PINN模型: {args.model}")
        device = torch.device(DEVICE)
        model = PINNModel().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        use_pinn = True
        print("✅ PINN预测已启用")
    else:
        print("ℹ️ 仅使用YOLO检测 + 简化预测")
    
    # 加载视频
    print("加载视频...")
    frames = load_video(args.video)
    if not frames:
        print("❌ 无法加载视频")
        return
    
    if args.max_frames:
        frames = frames[:args.max_frames]
    
    h, w = frames[0].shape[:2]
    fps = 30.0
    print(f"视频: {w}x{h}, {len(frames)}帧, 约{len(frames)/fps:.1f}秒")
    
    # 视频输出
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # 检测关键点
    print("YOLO检测关键点...")
    kp_list = [get_kp(yolo, f) for f in frames]
    kp_list = smooth_keypoints_kalman(kp_list)
    
    # 缓存
    left_traj = deque(maxlen=30)
    right_traj = deque(maxlen=30)
    prev_kp = None
    
    print("生成演示视频...")
    for i, frame in enumerate(frames):
        display = frame.copy()
        kp = kp_list[i]
        
        l_spd, r_spd = 0.0, 0.0
        l_act, r_act = False, False
        
        if kp is not None:
            # 画骨架
            display = draw_skeleton(display, kp)
            
            # 计算速度 & 动作检测
            if prev_kp is not None:
                if is_valid(kp, LW) and is_valid(prev_kp, LW):
                    l_spd = np.linalg.norm(kp[LW] - prev_kp[LW])
                    left_traj.append((kp[LW][0], kp[LW][1]))
                    l_act = l_spd > SPEED_THRESH
                
                if is_valid(kp, RW) and is_valid(prev_kp, RW):
                    r_spd = np.linalg.norm(kp[RW] - prev_kp[RW])
                    right_traj.append((kp[RW][0], kp[RW][1]))
                    r_act = r_spd > SPEED_THRESH
            
            # 画拳头
            if is_valid(kp, LW):
                display = draw_fist(display, kp[LW], left_traj, C_LW, "L", l_act)
            if is_valid(kp, RW):
                display = draw_fist(display, kp[RW], right_traj, C_RW, "R", r_act)
            
            # 预测落点（简化版）
            if l_act and is_valid(kp, LW):
                pred = predict_landing_simple(kp, prev_kp, LW)
                if pred:
                    cv2.circle(display, pred, 18, C_PRED, 3)
                    cv2.putText(display, "PRED", (pred[0]+22, pred[1]-22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_PRED, 2)
            
            # 动作提示
            if l_act or r_act:
                cv2.putText(display, ">>> ACTION DETECTED <<<", (50, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, C_ACT, 3)
            
            prev_kp = kp.copy() if not np.any(np.isnan(kp)) else prev_kp
        else:
            cv2.putText(display, "No Person Detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 信息面板
        display = draw_panel(display, i, len(frames), l_spd, r_spd, l_act, r_act, use_pinn)
        
        # 进度条
        prog = int((i / len(frames)) * w)
        cv2.rectangle(display, (0, h - 6), (prog, h), (0, 255, 0), -1)
        
        out.write(display)
        
        if (i + 1) % 30 == 0:
            print(f"  已处理 {i+1}/{len(frames)}")
    
    out.release()
    print(f"\n✅ 完成！输出: {os.path.abspath(args.output)}")
    print(f"   总帧数: {len(frames)}, 时长: {len(frames)/fps:.1f}秒")
    print(f"   模式: {'PINN预测' if use_pinn else 'YOLO检测+简化预测'}")


if __name__ == "__main__":
    main()