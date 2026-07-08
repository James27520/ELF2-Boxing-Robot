"""
losses.py
V8版本：只计算终点落点误差，0.01m精度目标
"""

import torch
from config import LOSS_WEIGHTS, FPS, MAX_PRED_RANGE, HUMAN_MAX_ACCEL, TARGET_MAE


def hybrid_loss(pred, gt, model_out):
    """
    V8：只预测1帧终点，损失函数只计算终点MAE
    pred: [B, 2, 3] - 2个手，每个手3维坐标
    gt: [B, 2, 3] - 同上
    """
    w = LOSS_WEIGHTS

    pred_l, pred_r = pred[:, 0], pred[:, 1]
    gt_l, gt_r = gt[:, 0], gt[:, 1]

    # 1. 终点MAE（核心损失）
    final_loss = (torch.mean(torch.abs(pred_l - gt_l)) +
                  torch.mean(torch.abs(pred_r - gt_r))) / 2

    # 2. 物理模型精度
    physics_l = model_out['physics_traj'][:, 0]
    physics_r = model_out['physics_traj'][:, 1]
    physics_loss = (torch.mean(torch.abs(physics_l - gt_l)) +
                    torch.mean(torch.abs(physics_r - gt_r))) / 2

    # 3. 残差限制
    res_l = model_out['residual'][:, 0]
    res_r = model_out['residual'][:, 1]
    residual_penalty = (torch.mean(torch.relu(torch.abs(res_l) - 0.1)) +
                        torch.mean(torch.relu(torch.abs(res_r) - 0.1))) / 2

    # 4. 权重正则化
    weights = model_out['weights']
    all_weights = list(weights['left'].values()) + list(weights['right'].values())
    weight_reg = sum(w ** 2 for w in all_weights) * 0.01

    # 5. XYZ分别损失（重点优化每个维度）
    loss_x = (torch.mean(torch.abs(pred_l[:, 0] - gt_l[:, 0])) +
              torch.mean(torch.abs(pred_r[:, 0] - gt_r[:, 0]))) / 2

    loss_y = (torch.mean(torch.abs(pred_l[:, 1] - gt_l[:, 1])) +
              torch.mean(torch.abs(pred_r[:, 1] - gt_r[:, 1]))) / 2

    loss_z = (torch.mean(torch.abs(pred_l[:, 2] - gt_l[:, 2])) +
              torch.mean(torch.abs(pred_r[:, 2] - gt_r[:, 2]))) / 2

    # 6. 输出范围惩罚
    out_of_range = torch.mean(torch.relu(torch.abs(pred) - MAX_PRED_RANGE))

    # V8：组合损失（重点优化终点精度）
    total = (
        0.40 * final_loss +        # 终点MAE（最高权重）
        0.10 * physics_loss +      # 物理模型
        0.05 * residual_penalty +  # 残差限制
        0.02 * weight_reg +        # 权重正则
        0.15 * loss_x +            # X维度
        0.15 * loss_y +            # Y维度
        0.20 * loss_z +            # Z维度（重点）
        0.03 * out_of_range        # 输出范围
    )

    return total, {
        'final': final_loss.item(),
        'physics': physics_loss.item(),
        'residual': residual_penalty.item(),
        'weight_reg': weight_reg,
        'x': loss_x.item(),
        'y': loss_y.item(),
        'z': loss_z.item(),
        'out_of_range': out_of_range.item(),
        'avg_mae': final_loss.item()
    }
