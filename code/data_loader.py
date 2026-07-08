"""
data_loader.py
V9版本：
1. 检测有效动作起始/结束帧（基于手腕速度）
2. 只对齐有效动作区间
3. 滑动窗口数据构建：任意10帧 -> 第11帧
"""

import os
import cv2
import numpy as np
import torch
from scipy.signal import correlate, find_peaks
from scipy.interpolate import interp1d
from ultralytics import YOLO

from config import *


def get_kp(model, frame):
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.xy) == 0:
        return None
    kp = res.keypoints.xy[0].cpu().numpy()
    if kp.shape[0] != JOINTS:
        return None
    if hasattr(res.keypoints, 'conf') and res.keypoints.conf is not None:
        conf = res.keypoints.conf[0].cpu().numpy()
        kp[conf < 0.3] = np.nan
    return kp


def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration': duration,
        'aspect_ratio': width / height if height > 0 else 0
    }


def resample_keypoints(kp_list, original_fps, target_fps=TARGET_FPS):
    if abs(original_fps - target_fps) < 0.5:
        if DEBUG_ALIGN:
            print(f"    帧率{original_fps:.1f}接近目标{target_fps}，跳过重采样")
        return kp_list

    if len(kp_list) == 0:
        return kp_list

    original_times = np.arange(len(kp_list)) / original_fps
    duration = original_times[-1]
    target_times = np.arange(0, duration, 1.0 / target_fps)

    if DEBUG_ALIGN:
        print(f"    重采样: {original_fps:.1f}fps -> {target_fps}fps")
        print(f"    原始帧数: {len(kp_list)} -> 目标帧数: {len(target_times)}")

    resampled_kp = []

    for j in range(JOINTS):
        for dim in range(2):
            valid_times = []
            valid_values = []
            for i, kp in enumerate(kp_list):
                if kp is not None and not np.isnan(kp[j, dim]):
                    valid_times.append(original_times[i])
                    valid_values.append(kp[j, dim])

            if len(valid_times) < 2:
                continue

            try:
                f_interp = interp1d(valid_times, valid_values, 
                                   kind='linear', 
                                   fill_value='extrapolate',
                                   bounds_error=False)
                interpolated = f_interp(target_times)

                for i, t in enumerate(target_times):
                    if i >= len(resampled_kp):
                        resampled_kp.append(np.full((JOINTS, 2), np.nan))
                    resampled_kp[i][j, dim] = interpolated[i]
            except Exception as e:
                if DEBUG_ALIGN:
                    print(f"    警告: 关节{j}维度{dim}插值失败: {e}")
                continue

    if DEBUG_ALIGN:
        print(f"    重采样完成: {len(resampled_kp)}帧")

    return resampled_kp


def is_key_joints_valid(kp):
    if kp is None:
        return False
    key_joints = [LS, RS, LE, RE, LW, RW, HIP_L, HIP_R]
    return not np.any(np.isnan(kp[key_joints]))


class KalmanFilter1D:
    def __init__(self, process_noise=KALMAN_PROCESS_NOISE, measurement_noise=KALMAN_MEASUREMENT_NOISE):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_noise
        self.R = measurement_noise
        self.initialized = False

    def update(self, measurement):
        if np.isnan(measurement):
            return self.x

        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x


def smooth_keypoints_kalman(kp_list):
    if len(kp_list) == 0:
        return kp_list

    filters = {}
    for j in range(JOINTS):
        for dim in range(2):
            filters[(j, dim)] = KalmanFilter1D()

    smoothed = []
    for kp in kp_list:
        if kp is None:
            smoothed.append(None)
            continue

        kp_smooth = kp.copy()
        for j in range(JOINTS):
            for dim in range(2):
                if not np.isnan(kp[j, dim]):
                    kp_smooth[j, dim] = filters[(j, dim)].update(kp[j, dim])
        smoothed.append(kp_smooth)

    return smoothed


def detect_action_boundaries(kp_list, fps=30):
    """
    V9核心改进：检测有效动作区间
    基于手腕速度检测动作开始和结束

    返回: (start_frame, end_frame) 或 (None, None) 如果未检测到
    """
    if len(kp_list) < MIN_ACTION_FRAMES:
        return None, None

    # 计算手腕速度
    wrist_speeds = []
    for i in range(len(kp_list)):
        if i == 0 or kp_list[i] is None or kp_list[i-1] is None:
            wrist_speeds.append(0.0)
            continue

        # 计算左右手腕速度
        v_l = np.linalg.norm(kp_list[i][LW] - kp_list[i-1][LW]) if not np.any(np.isnan(kp_list[i][LW])) and not np.any(np.isnan(kp_list[i-1][LW])) else 0
        v_r = np.linalg.norm(kp_list[i][RW] - kp_list[i-1][RW]) if not np.any(np.isnan(kp_list[i][RW])) and not np.any(np.isnan(kp_list[i-1][RW])) else 0
        wrist_speeds.append(max(v_l, v_r))

    wrist_speeds = np.array(wrist_speeds)

    # 平滑速度
    if len(wrist_speeds) >= 5:
        wrist_speeds = np.convolve(wrist_speeds, np.ones(5)/5, mode='same')

    # 检测动作开始：速度首次超过阈值
    action_frames = np.where(wrist_speeds > WRIST_SPEED_THRESHOLD)[0]

    if len(action_frames) == 0:
        print(f"  ⚠️ 未检测到动作（手腕速度始终<{WRIST_SPEED_THRESHOLD}）")
        return None, None

    # 找连续的动作区间
    start_frame = max(0, action_frames[0] - LEADING_BUFFER)
    end_frame = min(len(kp_list) - 1, action_frames[-1] + TRAILING_BUFFER)

    # 确保动作区间足够长
    if end_frame - start_frame < MIN_ACTION_FRAMES:
        print(f"  ⚠️ 动作区间太短({end_frame - start_frame}帧)，扩展中...")
        # 尝试扩展
        if start_frame > 0:
            start_frame = max(0, start_frame - (MIN_ACTION_FRAMES - (end_frame - start_frame)) // 2)
        if end_frame < len(kp_list) - 1:
            end_frame = min(len(kp_list) - 1, end_frame + (MIN_ACTION_FRAMES - (end_frame - start_frame)) // 2)

    print(f"  检测到动作区间: [{start_frame}, {end_frame}] ({end_frame - start_frame + 1}帧)")
    print(f"  最大手腕速度: {np.max(wrist_speeds):.1f}像素/帧")

    return start_frame, end_frame


def align_by_action(kp1_list, kp2_list, fps1=30, fps2=30, info1=None, info2=None):
    """
    V9改进版对齐：
    1. 先检测每个视频的有效动作区间
    2. 只截取有效区间进行对齐
    3. 使用动作强度相关对齐
    """
    # 帧率重采样
    if abs(fps1 - fps2) > 0.5 or abs(fps1 - TARGET_FPS) > 0.5 or abs(fps2 - TARGET_FPS) > 0.5:
        print(f"  检测到帧率差异: {fps1:.1f}fps vs {fps2:.1f}fps")
        print(f"  执行时间戳重采样...")
        kp1_list = resample_keypoints(kp1_list, fps1, TARGET_FPS)
        kp2_list = resample_keypoints(kp2_list, fps2, TARGET_FPS)
        print(f"  重采样后: video1={len(kp1_list)}帧, video2={len(kp2_list)}帧")

    # V9：检测有效动作区间
    if ACTION_DETECTION:
        print(f"  检测动作区间...")
        start1, end1 = detect_action_boundaries(kp1_list)
        start2, end2 = detect_action_boundaries(kp2_list)

        if start1 is None or start2 is None:
            print(f"  ⚠️ 无法检测动作区间，使用全视频对齐")
            start1, end1 = 0, len(kp1_list) - 1
            start2, end2 = 0, len(kp2_list) - 1

        # 截取有效区间
        kp1_valid = kp1_list[start1:end1+1]
        kp2_valid = kp2_list[start2:end2+1]

        print(f"  有效区间: video1[{start1}:{end1}] ({len(kp1_valid)}帧), video2[{start2}:{end2}] ({len(kp2_valid)}帧)")
    else:
        kp1_valid = kp1_list
        kp2_valid = kp2_list
        start1, start2 = 0, 0

    # 计算动作强度
    int1 = compute_action_intensity(kp1_valid)
    int2 = compute_action_intensity(kp2_valid)

    if int1.std() < 1e-6 or int2.std() < 1e-6:
        return [], [], 0, "动作强度过低", None, (start1, start2)

    int1 = np.convolve(int1, np.ones(5)/5, mode='same')
    int2 = np.convolve(int2, np.ones(5)/5, mode='same')

    int1 = (int1 - int1.mean()) / (int1.std() + 1e-6)
    int2 = (int2 - int2.mean()) / (int2.std() + 1e-6)

    corr = correlate(int1, int2, mode='full')
    lag = np.argmax(corr) - (len(int2) - 1)

    max_lag = min(len(kp1_valid), len(kp2_valid)) // 2
    if abs(lag) > max_lag:
        print(f"  ⚠️ 偏移{lag}超过限制({max_lag})，回退到0")
        lag = 0

    if lag >= 0:
        aligned1 = kp1_valid[lag:]
        aligned2 = kp2_valid[:len(aligned1)]
    else:
        aligned2 = kp2_valid[-lag:]
        aligned1 = kp1_valid[:len(aligned2)]

    min_len = min(len(aligned1), len(aligned2))
    aligned1 = aligned1[:min_len]
    aligned2 = aligned2[:min_len]

    match_score = np.max(corr) / (len(int2) + 1e-6)

    if match_score < 0.5:
        print(f"  动作强度匹配得分低({match_score:.2f})，尝试关键动作点对齐")
        peaks1 = find_action_peaks(int1)
        peaks2 = find_action_peaks(int2)
        if len(peaks1) > 0 and len(peaks2) > 0:
            peak_lag = peaks1[0] - peaks2[0]
            if abs(peak_lag) <= max_lag:
                lag = peak_lag
                if lag >= 0:
                    aligned1 = kp1_valid[lag:]
                    aligned2 = kp2_valid[:len(aligned1)]
                else:
                    aligned2 = kp2_valid[-lag:]
                    aligned1 = kp1_valid[:len(aligned2)]
                min_len = min(len(aligned1), len(aligned2))
                aligned1 = aligned1[:min_len]
                aligned2 = aligned2[:min_len]
                match_score = 0.6
                print(f"  用峰值对齐: 偏移{lag}帧")

    # 髋Y坐标验证
    if len(aligned1) > 10 and len(aligned2) > 10:
        hip_y1 = np.array([(k[HIP_L,1]+k[HIP_R,1])/2 for k in aligned1 if k is not None])
        hip_y2 = np.array([(k[HIP_L,1]+k[HIP_R,1])/2 for k in aligned2 if k is not None])
        if len(hip_y1) > 10 and len(hip_y2) > 10:
            hip_y1 = hip_y1 - hip_y1.mean()
            hip_y2 = hip_y2 - hip_y2.mean()
            if len(hip_y1) == len(hip_y2):
                corr_y = np.corrcoef(hip_y1, hip_y2)[0, 1]
                if abs(corr_y) > match_score:
                    match_score = abs(corr_y)
                    print(f"  髋Y坐标验证通过: 相关性{corr_y:.2f}")

    resolution_info = {
        'fps1': fps1, 'fps2': fps2,
        'width1': info1['width'] if info1 else 720,
        'height1': info1['height'] if info1 else 1280,
        'width2': info2['width'] if info2 else 544,
        'height2': info2['height'] if info2 else 1280,
    }

    # 返回原始偏移（用于调试）
    original_offsets = (start1, start2)

    return aligned1, aligned2, lag, f"动作匹配得分:{match_score:.2f}", resolution_info, original_offsets


def compute_action_intensity(kp_list):
    intensities = []
    for i in range(len(kp_list)):
        if kp_list[i] is None:
            intensities.append(0.0)
            continue
        kp = kp_list[i]
        if i == 0 or kp_list[i-1] is None:
            intensities.append(0.0)
            continue
        prev = kp_list[i-1]
        total_v = 0
        valid_count = 0
        for j in [LS, RS, LE, RE, LW, RW, HIP_L, HIP_R]:
            if not np.any(np.isnan(kp[j])) and not np.any(np.isnan(prev[j])):
                total_v += np.linalg.norm(kp[j] - prev[j])
                valid_count += 1
        intensities.append(total_v / max(valid_count, 1))
    return np.array(intensities)


def find_action_peaks(intensity, threshold=0.5):
    peaks, _ = find_peaks(intensity, height=threshold)
    return peaks


def fill_and_segment(aligned1, aligned2, max_fill=5):
    segments = []
    current1, current2 = [], []
    last_valid1, last_valid2 = None, None
    consecutive_fail = 0
    stats = {
        'original_frames': len(aligned1),
        'filled_frames': 0,
        'dropped_frames': 0,
        'breakpoints': []
    }

    for i in range(len(aligned1)):
        kp1, kp2 = aligned1[i], aligned2[i]
        valid1 = is_key_joints_valid(kp1)
        valid2 = is_key_joints_valid(kp2)

        if valid1 or valid2:
            stats['filled_frames'] += 1
        else:
            stats['dropped_frames'] += 1

        frame_added = False

        if valid1 and valid2:
            current1.append(kp1)
            current2.append(kp2)
            last_valid1, last_valid2 = kp1, kp2
            consecutive_fail = 0
            frame_added = True

        elif valid1 and not valid2:
            if last_valid2 is not None and consecutive_fail < max_fill:
                current1.append(kp1)
                current2.append(last_valid2.copy())
                last_valid1 = kp1
                consecutive_fail += 1
                frame_added = True
            else:
                consecutive_fail += 1

        elif not valid1 and valid2:
            if last_valid1 is not None and consecutive_fail < max_fill:
                current1.append(last_valid1.copy())
                current2.append(kp2)
                last_valid2 = kp2
                consecutive_fail += 1
                frame_added = True
            else:
                consecutive_fail += 1

        else:
            if last_valid1 is not None and last_valid2 is not None and consecutive_fail < max_fill:
                current1.append(last_valid1.copy())
                current2.append(last_valid2.copy())
                consecutive_fail += 1
                frame_added = True
            else:
                consecutive_fail += 1

        if consecutive_fail > max_fill and len(current1) > 0:
            stats['breakpoints'].append(len(current1))
            if len(current1) >= INPUT_T + PRED_T:
                segments.append((current1.copy(), current2.copy()))
            current1, current2 = [], []
            last_valid1, last_valid2 = None, None
            consecutive_fail = 0

    if len(current1) >= INPUT_T + PRED_T:
        segments.append((current1, current2))
    elif len(current1) > 0:
        stats['dropped_frames'] += len(current1)

    return segments, stats


def interpolate_nan_kp(kp):
    if kp is None:
        return None
    kp = kp.copy()

    left_joints = {5, 7, 9, 11, 13, 15}
    right_joints = {6, 8, 10, 12, 14, 16}

    for j in range(JOINTS):
        if np.any(np.isnan(kp[j])):
            if j in left_joints:
                side = 'left'
            elif j in right_joints:
                side = 'right'
            else:
                side = 'center'

            same_side_candidates = []
            all_candidates = []
            for k in range(JOINTS):
                if k == j or np.any(np.isnan(kp[k])):
                    continue
                k_side = 'left' if k in left_joints else 'right' if k in right_joints else 'center'
                dist = abs(k - j)
                candidate = (dist, kp[k])
                all_candidates.append(candidate)
                if side == 'center' or k_side == side or k_side == 'center':
                    same_side_candidates.append(candidate)

            if same_side_candidates:
                nearest = min(same_side_candidates, key=lambda x: x[0])
            elif all_candidates:
                nearest = min(all_candidates, key=lambda x: x[0])
            else:
                continue

            kp[j] = nearest[1]
    return kp


def compute_scale(front_kp, side_kp, frame_idx=0, resolution_info=None):
    front_kp = interpolate_nan_kp(front_kp)
    side_kp = interpolate_nan_kp(side_kp)
    if front_kp is None or side_kp is None:
        return None, None, 0.0

    if resolution_info:
        w1, h1 = resolution_info['width1'], resolution_info['height1']
        w2, h2 = resolution_info['width2'], resolution_info['height2']
    else:
        w1, h1 = 720, 1280
        w2, h2 = 544, 1280

    pixel_density_ratio = w2 / w1 if w1 > 0 and w2 > 0 else 1.0

    front_shoulder_px = np.linalg.norm(front_kp[LS] - front_kp[RS])
    if front_shoulder_px > 10 and front_shoulder_px < 500:
        scale_from_shoulder = SHOULDER_REAL / front_shoulder_px
    else:
        scale_from_shoulder = None

    side_height_px = abs(side_kp[NOSE, 1] - (side_kp[HIP_L, 1] + side_kp[HIP_R, 1]) / 2)
    if side_height_px > 20 and side_height_px < 800:
        scale_from_height = HEIGHT_REAL / side_height_px
    else:
        scale_from_height = None

    hip_width_px = np.linalg.norm(front_kp[HIP_L] - front_kp[HIP_R])
    if hip_width_px > 5 and hip_width_px < 300:
        scale_from_hip = HIP_WIDTH_REAL / hip_width_px
    else:
        scale_from_hip = None

    scales = [s for s in [scale_from_shoulder, scale_from_height, scale_from_hip] if s is not None]

    if len(scales) == 0:
        default_scale = 0.41 / 150.0
        print(f"  [帧{frame_idx}] 所有尺度计算失败，使用默认值: {default_scale:.6f}")
        return default_scale, default_scale * pixel_density_ratio, 0.3

    if len(scales) >= 2:
        median_scale = np.median(scales)
        std_scale = np.std(scales)
        confidence = 1.0 - min(std_scale / median_scale, 0.5)
    else:
        median_scale = scales[0]
        confidence = 0.5

    front_scale = median_scale
    side_scale = median_scale * pixel_density_ratio

    if DEBUG_SCALE:
        shoulder_str = f"{scale_from_shoulder:.6f}" if scale_from_shoulder else "N/A"
        height_str = f"{scale_from_height:.6f}" if scale_from_height else "N/A"
        hip_str = f"{scale_from_hip:.6f}" if scale_from_hip else "N/A"
        print(f"  [帧{frame_idx}] 尺度:")
        print(f"    分辨率: 正面{w1}x{h1}, 侧面{w2}x{h2}")
        print(f"    像素密度比: {pixel_density_ratio:.3f}")
        print(f"    肩宽: {shoulder_str}, 身高: {height_str}, 髋宽: {hip_str}")
        print(f"    中值: {median_scale:.6f}, 正面: {front_scale:.6f}, 侧面: {side_scale:.6f}")
        print(f"    置信度: {confidence:.2f}")

    if confidence < 0.5:
        print(f"  [帧{frame_idx}] ⚠️ 尺度置信度低({confidence:.2f})，使用默认值")
        front_scale = 0.41 / 150.0
        side_scale = front_scale * pixel_density_ratio

    return front_scale, side_scale, confidence


def reconstruct_3d(front_kp, side_kp, scale_front=None, scale_side=None, 
                   frame_idx=0, debug=False, resolution_info=None):
    front_kp = interpolate_nan_kp(front_kp) if front_kp is not None else None
    side_kp = interpolate_nan_kp(side_kp) if side_kp is not None else None
    if front_kp is None or side_kp is None:
        return None

    if scale_front is None or scale_side is None:
        scale_front, scale_side, confidence = compute_scale(front_kp, side_kp, frame_idx, resolution_info)
        if scale_front is None:
            return None

    hip_f = front_kp[HIP_L]
    hip_s = side_kp[HIP_L]

    x = (front_kp[:, 0] - hip_f[0]) * scale_front
    y = ((front_kp[:, 1] - hip_f[1]) * scale_front + 
         (side_kp[:, 1] - hip_s[1]) * scale_side) / 2
    z = (side_kp[:, 0] - hip_s[0]) * scale_side

    coords = np.stack([x, y, z], axis=-1)

    for dim, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        dim_min, dim_max = np.min(coords[:, dim]), np.max(coords[:, dim])
        dim_range = dim_max - dim_min
        if dim_range > 4.0 or dim_min < -3.0 or dim_max > 3.0:
            print(f"  [帧{frame_idx}] 警告: {name}轴范围异常 [{dim_min:.2f}, {dim_max:.2f}]m")

    return coords


def normalize(seq):
    hip = (seq[:, HIP_L] + seq[:, HIP_R]) / 2
    seq = seq - hip[:, None, :]
    return seq


_feature_mean = None
_feature_std = None


def normalize_features(features):
    global _feature_mean, _feature_std

    if _feature_mean is None or _feature_std is None:
        _feature_mean = np.mean(features, axis=(0, 1, 2), keepdims=True)
        _feature_std = np.std(features, axis=(0, 1, 2), keepdims=True) + 1e-6

    return (features - _feature_mean) / _feature_std


def reset_feature_norm():
    global _feature_mean, _feature_std
    _feature_mean = None
    _feature_std = None


def build_dataset(model, data_dirs=None, is_train=True):
    if data_dirs is None:
        data_dirs = DATA_DIRS

    X, Y = [], []
    total_stats = {
        'original_frames': 0,
        'filled_frames': 0,
        'dropped_frames': 0,
        'segments': 0
    }

    for d in data_dirs:
        if not os.path.exists(d):
            print(f"警告: 目录 {d} 不存在，跳过")
            continue

        v1 = os.path.join(d, "video1.mp4")
        v2 = os.path.join(d, "video2.mp4")

        if not os.path.exists(v1) or not os.path.exists(v2):
            continue

        print(f"\n{'='*60}")
        print(f"处理目录: {d}")
        print(f"{'='*60}")

        info1 = get_video_info(v1)
        info2 = get_video_info(v2)

        print(f"Video1: {info1['width']}x{info1['height']}, {info1['fps']:.1f}fps, {info1['total_frames']}帧")
        print(f"Video2: {info2['width']}x{info2['height']}, {info2['fps']:.1f}fps, {info2['total_frames']}帧")

        if info1['width'] != info2['width'] or info1['height'] != info2['height']:
            print(f"  ⚠️ 分辨率不同！将使用分辨率感知尺度计算")

        f1 = load_video(v1)
        f2 = load_video(v2)
        print(f"加载帧数: video1={len(f1)}帧, video2={len(f2)}帧")

        kp1_list = [get_kp(model, f) for f in f1]
        kp2_list = [get_kp(model, f) for f in f2]

        print(f"  应用卡尔曼滤波...")
        kp1_list = smooth_keypoints_kalman(kp1_list)
        kp2_list = smooth_keypoints_kalman(kp2_list)

        valid1 = sum(1 for k in kp1_list if is_key_joints_valid(k))
        valid2 = sum(1 for k in kp2_list if is_key_joints_valid(k))
        print(f"有效帧: front={valid1}/{len(f1)}({valid1/len(f1)*100:.1f}%), side={valid2}/{len(f2)}({valid2/len(f2)*100:.1f}%)")

        # V9：改进对齐（检测动作区间）
        aligned1, aligned2, lag, align_msg, resolution_info, original_offsets = align_by_action(
            kp1_list, kp2_list, 
            fps1=info1['fps'], fps2=info2['fps'],
            info1=info1, info2=info2
        )
        print(f"对齐: 偏移{lag}帧, {align_msg}")
        print(f"对齐后: {len(aligned1)}帧")

        if len(aligned1) == 0:
            print(f"  警告: {d} 对齐失败")
            continue

        segments, stats = fill_and_segment(aligned1, aligned2)

        print(f"统计: 原始={stats['original_frames']}, 补齐={stats['filled_frames']}, 丢弃={stats['dropped_frames']}")
        print(f"生成 {len(segments)} 个有效段")

        total_stats['original_frames'] += stats['original_frames']
        total_stats['filled_frames'] += stats['filled_frames']
        total_stats['dropped_frames'] += stats['dropped_frames']

        if len(segments) == 0:
            print(f"  警告: {d} 没有有效段")
            continue

        from physics_features import compute_physics
        for seg_idx, (seg1, seg2) in enumerate(segments):
            if len(seg1) < INPUT_T + PRED_T:
                continue

            seg1_clean = [interpolate_nan_kp(k) for k in seg1]
            seg2_clean = [interpolate_nan_kp(k) for k in seg2]

            scales_front = []
            scales_side = []
            confidences = []
            for i, (k1, k2) in enumerate(zip(seg1_clean, seg2_clean)):
                sf, ss, conf = compute_scale(k1, k2, frame_idx=i, resolution_info=resolution_info)
                if sf is not None:
                    scales_front.append(sf)
                    scales_side.append(ss)
                    confidences.append(conf)
                else:
                    if scales_front:
                        scales_front.append(scales_front[-1])
                        scales_side.append(scales_side[-1])
                        confidences.append(confidences[-1])
                    else:
                        default = 0.41 / 150.0
                        if resolution_info:
                            ratio = resolution_info['width2'] / resolution_info['width1']
                        else:
                            ratio = 1.0
                        scales_front.append(default)
                        scales_side.append(default * ratio)
                        confidences.append(0.3)

            if len(scales_front) >= SCALE_SMOOTH_WINDOW:
                kernel = np.ones(SCALE_SMOOTH_WINDOW) / SCALE_SMOOTH_WINDOW
                scales_front = np.convolve(scales_front, kernel, mode='same')
                scales_side = np.convolve(scales_side, kernel, mode='same')

            seq = []
            for i, (k1, k2) in enumerate(zip(seg1_clean, seg2_clean)):
                coords = reconstruct_3d(k1, k2, scales_front[i], scales_side[i], 
                                       frame_idx=i, debug=True, resolution_info=resolution_info)
                if coords is not None:
                    seq.append(coords)

            if len(seq) < INPUT_T + PRED_T:
                continue

            seq = np.array(seq)

            print(f"  段{seg_idx+1} 3D范围:")
            for dim, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
                dim_min, dim_max = np.min(seq[:, :, dim]), np.max(seq[:, :, dim])
                print(f"    {name}: [{dim_min:.3f}, {dim_max:.3f}]m")

            features = compute_physics(seq)
            seq = normalize(seq)

            # V9：滑动窗口数据构建
            # 任意连续10帧 -> 第11帧（终点落点）
            n_samples = 0
            for i in range(len(seq) - INPUT_T - PRED_T + 1):
                X.append(features[i:i + INPUT_T])
                y = seq[i + INPUT_T + PRED_T - 1, TARGET_JOINTS]  # [2, 3]
                Y.append(y)
                n_samples += 1

            total_stats['segments'] += 1
            print(f"  段{seg_idx+1}: {len(seg1)}帧, 生成 {n_samples} 个滑动窗口样本")

    if len(X) == 0:
        raise ValueError("没有构建出任何样本！请检查视频路径和格式。")

    X = np.array(X)
    Y = np.array(Y)

    if is_train:
        reset_feature_norm()
    X = normalize_features(X)

    print(f"\n{'='*60}")
    print("数据集构建完成")
    print(f"  总样本数: {len(X)}")
    print(f"  特征范围: [{np.min(X):.3f}, {np.max(X):.3f}]")
    print(f"{'='*60}")

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))
