"""
train.py
V8版本 + 新增：保存模型时自动导出公式系数
原有代码完全保留，只在保存模型后追加3行
"""

import os
import numpy as np
import torch
import torch.optim as optim
from config import *
from models import PINNModel
from losses import hybrid_loss
from metrics import evaluate


def random_mask_input(x, min_len=MIN_INPUT_T, max_len=MAX_INPUT_T):
    B, T = x.shape[0], x.shape[1]
    valid_lens = torch.randint(min_len, max_len + 1, (B,))
    x_masked = x.clone()
    for i in range(B):
        valid_len = valid_lens[i].item()
        if valid_len < T:
            x_masked[i, :T - valid_len] = 0
    return x_masked


def train_model(X_train, Y_train, X_test, Y_test, fold_idx=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    net = PINNModel().to(DEVICE)

    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, 
                            betas=(0.9, 0.999))

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=20
    )

    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=LR/100
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[20]
    )

    weights_history = []
    best_mae = float('inf')
    patience_counter = 0
    best_epoch = 0

    # V8：跟踪每个维度的最佳MAE
    best_dim_mae = {
        'left_x': float('inf'), 'left_y': float('inf'), 'left_z': float('inf'),
        'right_x': float('inf'), 'right_y': float('inf'), 'right_z': float('inf')
    }

    prefix = f"Fold {fold_idx} " if fold_idx is not None else ""
    print(f"\n{'='*60}")
    print(f"{prefix}开始训练 PINN V8")
    print(f"输入: {MIN_INPUT_T}-{MAX_INPUT_T}帧")
    print(f"目标: 终点MAE < {TARGET_MAE}m (1厘米)")
    print(f"{'='*60}\n")

    for epoch in range(MAX_EPOCHS):
        net.train()

        x_train_masked = random_mask_input(X_train)

        pred, model_out = net(x_train_masked)
        loss, loss_dict = hybrid_loss(pred, Y_train, model_out)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n警告: Epoch {epoch+1} 损失异常，中断")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            weights_history.append(model_out['weights'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            net.eval()
            with torch.no_grad():
                pred_test, _ = net(X_test)

                if BASELINE_EVAL:
                    physics_pred = net.physics_baseline(X_test)
                    metrics_physics = evaluate(physics_pred, Y_test, f"{prefix}Epoch {epoch+1} [物理]")

                metrics = evaluate(pred_test, Y_test, f"{prefix}Epoch {epoch+1} [融合]")

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Loss={loss.item():.4f} | LR={current_lr:.6f} | "
                      f"MAE={loss_dict['avg_mae']:.4f}m")

                w = model_out['weights']
                print(f"左: pos={w['left']['pos']:.2f} vel={w['left']['vel']:.2f} acc={w['left']['acc']:.2f}")
                print(f"右: pos={w['right']['pos']:.2f} vel={w['right']['vel']:.2f} acc={w['right']['acc']:.2f}")

                if BASELINE_EVAL:
                    nn_mae = metrics['avg_mae']
                    phy_mae = metrics_physics['avg_mae']
                    if phy_mae > 0:
                        improvement = (phy_mae - nn_mae) / phy_mae * 100
                        print(f"  📊 融合vs物理: MAE={nn_mae:.4f}m vs {phy_mae:.4f}m (提升{improvement:.1f}%)")

                m = metrics
                # V8：检查每个维度的MAE是否<0.01m
                dim_status = {
                    'left_x': m['left']['mae_x'] < TARGET_MAE,
                    'left_y': m['left']['mae_y'] < TARGET_MAE,
                    'left_z': m['left']['mae_z'] < TARGET_MAE,
                    'right_x': m['right']['mae_x'] < TARGET_MAE,
                    'right_y': m['right']['mae_y'] < TARGET_MAE,
                    'right_z': m['right']['mae_z'] < TARGET_MAE,
                }

                for dim, status in dim_status.items():
                    if status:
                        hand, axis = dim.split('_')
                        current_mae = m[hand]['mae_' + axis]
                        if current_mae < best_dim_mae[dim]:
                            best_dim_mae[dim] = current_mae

                status_str = " | ".join([
                    f"{dim}: {'✓' if status else '✗'} ({m[dim.split('_')[0]]['mae_' + dim.split('_')[1]]:.4f}m)"
                    for dim, status in dim_status.items()
                ])
                print(f"  达标: {status_str}")

                # V8：停止条件：所有6个维度MAE<0.01m
                if all(dim_status.values()):
                    print(f"\n{'='*60}")
                    print(f"🎯 {prefix}全部达标！终点MAE<0.01m（Epoch {epoch+1}）")
                    print(f"  左手: X={m['left']['mae_x']:.4f}m Y={m['left']['mae_y']:.4f}m Z={m['left']['mae_z']:.4f}m")
                    print(f"  右手: X={m['right']['mae_x']:.4f}m Y={m['right']['mae_y']:.4f}m Z={m['right']['mae_z']:.4f}m")
                    print(f"{'='*60}")
                    break

                if metrics['avg_mae'] < best_mae:
                    best_mae = metrics['avg_mae']
                    best_epoch = epoch + 1
                    patience_counter = 0
                    model_name = f"best_model_fold{fold_idx}.pt" if fold_idx is not None else "best_model.pt"
                    torch.save(net.state_dict(), os.path.join(SAVE_DIR, model_name))
                    print(f"  💾 保存最佳模型 (MAE={best_mae:.4f}m)")
                    # ==================== 新增：导出公式 ====================
                    net.print_formula('left')
                    net.print_formula('right')
                    net.export_formula(os.path.join(SAVE_DIR, "formula.json"))
                    # ======================================================
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\n早停：{PATIENCE}轮未改善 (最佳MAE={best_mae:.4f}m @ Epoch {best_epoch})")
                        print(f"  最佳维度: {best_dim_mae}")
                        break

    model_name = f"best_model_fold{fold_idx}.pt" if fold_idx is not None else "best_model.pt"
    model_path = os.path.join(SAVE_DIR, model_name)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print(f"已加载最佳模型: {model_name}")

    return net, weights_history