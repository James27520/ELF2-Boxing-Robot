"""
visualization.py
V8版本：只显示终点落点
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_results(pred, gt, weights_history, save_path="outputs/results.png", physics_pred=None):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if physics_pred is not None and isinstance(physics_pred, torch.Tensor):
        physics_pred = physics_pred.detach().cpu().numpy()

    pred_l, pred_r = pred[:, 0], pred[:, 1]
    gt_l, gt_r = gt[:, 0], gt[:, 1]
    diff_l = np.abs(pred_l - gt_l)
    diff_r = np.abs(pred_r - gt_r)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    dims = ['X', 'Y', 'Z']
    for dim_idx, dim_name in enumerate(dims):
        # 误差分布
        ax = axes[0, dim_idx]
        errors = np.concatenate([diff_l[:, dim_idx], diff_r[:, dim_idx]])
        ax.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'均值={np.mean(errors):.4f}m')
        ax.set_title(f'{dim_name}维度 误差分布')
        ax.set_xlabel('绝对误差 (m)')
        ax.set_ylabel('频数')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 偏差
        ax = axes[1, dim_idx]
        bias_l = pred_l[:, dim_idx] - gt_l[:, dim_idx]
        bias_r = pred_r[:, dim_idx] - gt_r[:, dim_idx]
        ax.scatter(bias_l, bias_r, alpha=0.5, s=20)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f'{dim_name}维度 偏差散点')
        ax.set_xlabel('左手偏差 (m)')
        ax.set_ylabel('右手偏差 (m)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"结果图已保存: {save_path}")

    # 权重学习过程
    if len(weights_history) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(len(weights_history))
        for hand in ['left', 'right']:
            for key in weights_history[0][hand].keys():
                vals = [w[hand][key] for w in weights_history]
                label = f"{'左' if hand=='left' else '右'}手_{key}"
                ax.plot(epochs, vals, '-o', label=label, linewidth=2, markersize=3)
        ax.set_title('物理权重学习过程')
        ax.set_xlabel('记录点')
        ax.set_ylabel('权重')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        weight_path = save_path.replace('results', 'weights')
        plt.savefig(weight_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"权重图已保存: {weight_path}")

    # 3D落点图
    fig = plt.figure(figsize=(12, 5))

    for hand_idx, hand_name in [(0, '左手'), (1, '右手')]:
        ax = fig.add_subplot(1, 2, hand_idx+1, projection='3d')
        p = pred[:, hand_idx]
        g = gt[:, hand_idx]

        ax.scatter(g[:, 0], g[:, 1], g[:, 2], c='green', s=50, marker='o', label='真实', alpha=0.6)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='red', s=50, marker='x', label='预测', alpha=0.6)

        # 画连接线
        for i in range(min(len(p), 50)):  # 只画前50个避免混乱
            ax.plot([g[i, 0], p[i, 0]], [g[i, 1], p[i, 1]], [g[i, 2], p[i, 2]], 
                   'gray', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{hand_name} 落点对比')
        ax.legend()

    plt.tight_layout()
    traj_path = save_path.replace('results', '3d_landing')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"3D落点图已保存: {traj_path}")


def plot_3d_trajectory(pred, gt, save_path="outputs/3d_trajectory.png"):
    """V8：简化为只显示终点落点"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 5))

    for hand_idx, hand_name in [(0, '左手'), (1, '右手')]:
        ax = fig.add_subplot(1, 2, hand_idx+1, projection='3d')
        p = pred[:, hand_idx]
        g = gt[:, hand_idx]

        # 找误差最大的样本
        err = np.linalg.norm(p - g, axis=1)
        worst = np.argmax(err)

        ax.scatter(g[worst, 0], g[worst, 1], g[worst, 2], c='green', s=200, marker='*', label='真实落点')
        ax.scatter(p[worst, 0], p[worst, 1], p[worst, 2], c='red', s=200, marker='X', label='预测落点')
        ax.plot([g[worst, 0], p[worst, 0]], [g[worst, 1], p[worst, 1]], [g[worst, 2], p[worst, 2]], 
               'blue', linewidth=2, label=f'误差={err[worst]:.3f}m')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{hand_name} 最大误差样本')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"3D图已保存: {save_path}")
