"""
metrics.py
V8修复版：修复维度索引
"""

import numpy as np
import torch
from config import TARGET_MAE, HUMAN_MAX_ACCEL, HUMAN_MAX_SPEED, FPS


def evaluate_hand(pred, gt, hand_name):
    """
    V8：pred和gt都是 [B, 3]（只预测终点1帧）
    """
    diff = np.abs(pred - gt)  # [B, 3]

    mae_x = np.mean(diff[:, 0])
    mae_y = np.mean(diff[:, 1])
    mae_z = np.mean(diff[:, 2])

    bias_x = np.mean(pred[:, 0] - gt[:, 0])
    bias_y = np.mean(pred[:, 1] - gt[:, 1])
    bias_z = np.mean(pred[:, 2] - gt[:, 2])

    # 终点误差（欧氏距离）
    end_err = np.linalg.norm(pred - gt, axis=-1)
    end_mae = np.mean(end_err)

    print(f"\n  [{hand_name}]")
    print(f"  MAE: X={mae_x:.4f}m Y={mae_y:.4f}m Z={mae_z:.4f}m | 终点={end_mae:.4f}m")
    print(f"  偏差: X={bias_x:+.4f}m Y={bias_y:+.4f}m Z={bias_z:+.4f}m")
    print(f"  达标: X={'✓' if mae_x < TARGET_MAE else '✗'}  "
          f"Y={'✓' if mae_y < TARGET_MAE else '✗'}  "
          f"Z={'✓' if mae_z < TARGET_MAE else '✗'}")

    return {
        'mae_x': mae_x, 'mae_y': mae_y, 'mae_z': mae_z,
        'bias_x': bias_x, 'bias_y': bias_y, 'bias_z': bias_z,
        'end_mae': end_mae,
        'diff': diff,
    }


def evaluate(pred, gt, prefix=""):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    # V8：pred和gt都是 [B, 2, 3]
    pred_l, pred_r = pred[:, 0], pred[:, 1]
    gt_l, gt_r = gt[:, 0], gt[:, 1]

    print(f"\n{'='*60}")
    print(f"{prefix} 评估")
    print(f"{'='*60}")

    metrics_l = evaluate_hand(pred_l, gt_l, "左手")
    metrics_r = evaluate_hand(pred_r, gt_r, "右手")

    avg_mae = (metrics_l['mae_x'] + metrics_l['mae_y'] + metrics_l['mae_z'] +
               metrics_r['mae_x'] + metrics_r['mae_y'] + metrics_r['mae_z']) / 6

    print(f"\n  平均MAE: {avg_mae:.4f}m (目标<{TARGET_MAE}m)")
    print(f"{'='*60}")

    return {
        'left': metrics_l,
        'right': metrics_r,
        'avg_mae': avg_mae
    }