"""
physics_features.py
V6最终版：加速度截断，中心差分，人体运动学约束
"""

import numpy as np
from config import *


def compute_physics(seq):
    T = len(seq)

    if np.any(np.isnan(seq)):
        print(f"  警告: 检测到{np.sum(np.isnan(seq))}个nan值，进行插值")
        for j in range(JOINTS):
            for dim in range(3):
                if np.any(np.isnan(seq[:, j, dim])):
                    valid = ~np.isnan(seq[:, j, dim])
                    if np.any(valid):
                        seq[:, j, dim] = np.interp(
                            np.arange(T),
                            np.where(valid)[0],
                            seq[valid, j, dim]
                        )

    # 速度（中心差分）
    v = np.zeros_like(seq)
    v[1:-1] = (seq[2:] - seq[:-2]) / 2.0 * FPS
    v[0] = (seq[1] - seq[0]) * FPS
    v[-1] = (seq[-1] - seq[-2]) * FPS

    # 加速度
    a = np.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / 2.0 * FPS
    a[0] = (v[1] - v[0]) * FPS
    a[-1] = (v[-1] - v[-2]) * FPS

    # 加速度截断
    max_accel_detected = np.max(np.linalg.norm(a[:, [LW, RW]], axis=-1))
    if max_accel_detected > HUMAN_MAX_ACCEL * 5:
        print(f"  ⚠️ 极端加速度: {max_accel_detected:.1f}m/s²，截断")
        a = np.clip(a, -HUMAN_MAX_ACCEL * 2, HUMAN_MAX_ACCEL * 2)
    elif max_accel_detected > HUMAN_MAX_ACCEL:
        if DEBUG_ACCEL:
            print(f"  ⚠️ 高加速度: {max_accel_detected:.1f}m/s² > {HUMAN_MAX_ACCEL}m/s²")

    # 关节角度
    angles = np.zeros((T, 8))

    for t in range(T):
        upper_l = seq[t, LS] - seq[t, LE]
        fore_l = seq[t, LW] - seq[t, LE]
        cos_el = np.dot(upper_l, fore_l) / (np.linalg.norm(upper_l) * np.linalg.norm(fore_l) + 1e-6)
        angles[t, 0] = np.arccos(np.clip(cos_el, -1, 1))

        upper_r = seq[t, RS] - seq[t, RE]
        fore_r = seq[t, RW] - seq[t, RE]
        cos_er = np.dot(upper_r, fore_r) / (np.linalg.norm(upper_r) * np.linalg.norm(fore_r) + 1e-6)
        angles[t, 1] = np.arccos(np.clip(cos_er, -1, 1))

        torso_l = seq[t, HIP_L] - seq[t, LS]
        arm_l = seq[t, LE] - seq[t, LS]
        cos_sl = np.dot(torso_l, arm_l) / (np.linalg.norm(torso_l) * np.linalg.norm(arm_l) + 1e-6)
        angles[t, 2] = np.arccos(np.clip(cos_sl, -1, 1))

        torso_r = seq[t, HIP_R] - seq[t, RS]
        arm_r = seq[t, RE] - seq[t, RS]
        cos_sr = np.dot(torso_r, arm_r) / (np.linalg.norm(torso_r) * np.linalg.norm(arm_r) + 1e-6)
        angles[t, 3] = np.arccos(np.clip(cos_sr, -1, 1))

        wrist_v_l = v[t, LW]
        if np.linalg.norm(wrist_v_l) > 1e-6:
            angles[t, 4] = np.arctan2(wrist_v_l[2], np.linalg.norm(wrist_v_l[:2]))
            angles[t, 5] = np.arctan2(wrist_v_l[1], wrist_v_l[0])

        wrist_v_r = v[t, RW]
        if np.linalg.norm(wrist_v_r) > 1e-6:
            angles[t, 6] = np.arctan2(wrist_v_r[2], np.linalg.norm(wrist_v_r[:2]))
            angles[t, 7] = np.arctan2(wrist_v_r[1], wrist_v_r[0])

    # 角速度
    omega = np.zeros_like(angles)
    omega[1:-1] = (angles[2:] - angles[:-2]) / 2.0 * FPS
    omega[0] = (angles[1] - angles[0]) * FPS
    omega[-1] = (angles[-1] - angles[-2]) * FPS

    # 角加速度
    alpha = np.zeros_like(omega)
    alpha[1:-1] = (omega[2:] - omega[:-2]) / 2.0 * FPS
    alpha[0] = (omega[1] - omega[0]) * FPS
    alpha[-1] = (omega[-1] - omega[-2]) * FPS

    # 冲量
    MASS_FIST = 0.5
    impulse = np.zeros((T, 2))
    for t in range(1, T):
        impulse[t, 0] = np.linalg.norm((v[t, LW] - v[t-1, LW]) * MASS_FIST, axis=-1)
        impulse[t, 1] = np.linalg.norm((v[t, RW] - v[t-1, RW]) * MASS_FIST, axis=-1)

    # 组合特征 [T, 17, 21]
    features = np.zeros((T, JOINTS, 21))
    for t in range(T):
        for j in range(JOINTS):
            features[t, j, 0:3] = seq[t, j]
            features[t, j, 3:6] = v[t, j]
            features[t, j, 6:9] = a[t, j]
            features[t, j, 9:17] = angles[t]
            features[t, j, 17] = np.linalg.norm(omega[t, 4:6])
            features[t, j, 18] = np.linalg.norm(omega[t, 6:8])
            features[t, j, 19] = impulse[t, 0]
            features[t, j, 20] = impulse[t, 1]

    # 运动学检查
    max_speed = np.max(np.linalg.norm(v[:, [LW, RW]], axis=-1))
    max_accel = np.max(np.linalg.norm(a[:, [LW, RW]], axis=-1))
    if max_speed > HUMAN_MAX_SPEED:
        print(f"  ⚠️ 速度 {max_speed:.1f}m/s > 极限 {HUMAN_MAX_SPEED}m/s")
    if max_accel > HUMAN_MAX_ACCEL:
        print(f"  ⚠️ 加速度 {max_accel:.1f}m/s² > 极限 {HUMAN_MAX_ACCEL}m/s²")

    if np.any(np.isnan(features)):
        print(f"  警告: 特征中仍有{np.sum(np.isnan(features))}个nan，已置零")
        features = np.nan_to_num(features, nan=0.0)

    return features
