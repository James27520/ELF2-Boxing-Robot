"""
main.py
V8版本 + 新增：训练后打印公式、演示参数修改
原有代码完全保留，只在评估后追加公式输出
"""

import os
import numpy as np
import torch

from config import DEVICE, DATA_DIRS, SAVE_DIR, CROSS_VAL, N_FOLDS, BASELINE_EVAL
from data_loader import build_dataset, reset_feature_norm
from models import PINNModel
from train import train_model
from metrics import evaluate
from visualization import plot_results, plot_3d_trajectory


def main():
    print("=" * 70)
    print("拳击落点预测系统 - PINN V9")
    print("核心改进：")
    print("  1. 改进对齐：检测有效动作区间")
    print("  2. 滑动窗口：任意10帧 -> 第11帧")
    print("  3. 只预测终点落点（1帧）")
    print("=" * 70)

    from ultralytics import YOLO
    print("\n[1/5] 加载YOLO...")
    yolo = YOLO("yolov8n-pose.pt")

    if CROSS_VAL and len(DATA_DIRS) >= 2:
        print(f"\n[2/5] {N_FOLDS}折交叉验证")
        all_metrics = []

        for fold, test_dir in enumerate(DATA_DIRS):
            train_dirs = [d for d in DATA_DIRS if d != test_dir]
            print(f"\n{'='*60}")
            print(f"Fold {fold+1}/{N_FOLDS}: 训练={train_dirs}, 测试={test_dir}")
            print(f"{'='*60}")

            reset_feature_norm()

            print(f"\n构建训练集...")
            X_train, Y_train = build_dataset(yolo, train_dirs, is_train=True)
            # ========== 新增：保存特征统计量（部署用）==========
            from data_loader import _feature_mean, _feature_std
            import numpy as np
            if _feature_mean is not None and _feature_std is not None:
             np.savez(os.path.join(SAVE_DIR, "feature_stats.npz"),
                      mean=_feature_mean.squeeze(), std=_feature_std.squeeze())
             print(f"✓ 特征统计量已保存: {SAVE_DIR}/feature_stats.npz")
            # ==================================================
            print(f"构建测试集...")
            X_test, Y_test = build_dataset(yolo, [test_dir], is_train=False)

            print(f"\n[3/5] 数据划分...")
            X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
            X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
            print(f"训练: {len(X_train)} | 测试: {len(X_test)}")
            print(f"输入形状: {X_train.shape} | 输出形状: {Y_train.shape}")

            print(f"\n[4/5] 训练...")
            model, weights_history = train_model(X_train, Y_train, X_test, Y_test, fold_idx=fold)

            print(f"\n[5/5] Fold {fold+1} 评估...")
            model.eval()
            with torch.no_grad():
                pred_final, _ = model(X_test)

                physics_pred = None
                if BASELINE_EVAL:
                    physics_pred = model.physics_baseline(X_test)
                    evaluate(physics_pred, Y_test, f"Fold {fold+1} [物理]")

                final_metrics = evaluate(pred_final, Y_test, f"Fold {fold+1} [融合]")
                all_metrics.append(final_metrics)

                print(f"\n可视化...")
                plot_results(pred_final, Y_test, weights_history,
                           save_path=os.path.join(SAVE_DIR, f"results_fold{fold}.png"),
                           physics_pred=physics_pred)
                plot_3d_trajectory(pred_final, Y_test,
                                  save_path=os.path.join(SAVE_DIR, f"3d_landing_fold{fold}.png"))

        print(f"\n{'='*60}")
        print("交叉验证汇总")
        avg_mae = np.mean([m['avg_mae'] for m in all_metrics])
        std_mae = np.std([m['avg_mae'] for m in all_metrics])
        print(f"平均MAE: {avg_mae:.4f}m ± {std_mae:.4f}m")
        for i, m in enumerate(all_metrics):
            print(f"  Fold {i+1}: MAE={m['avg_mae']:.4f}m")
        print(f"{'='*60}")

    else:
        print("\n[2/5] 构建数据集...")
        reset_feature_norm()
        X, Y = build_dataset(yolo, DATA_DIRS, is_train=True)
        # ========== 新增：保存特征统计量（部署用）==========
        from data_loader import _feature_mean, _feature_std
        import numpy as np
        if _feature_mean is not None and _feature_std is not None:
         np.savez(os.path.join(SAVE_DIR, "feature_stats.npz"),
             mean=_feature_mean.squeeze(), std=_feature_std.squeeze())
         print(f"✓ 特征统计量已保存: {SAVE_DIR}/feature_stats.npz")
        # ==================================================
        print(f"总样本: {len(X)}")
        print(f"输入形状: {X.shape} | 输出形状: {Y.shape}")

        print("\n[3/5] 划分...")
        n_train = int(0.8 * len(X))
        idx = torch.randperm(len(X))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_train, Y_train = X[train_idx].to(DEVICE), Y[train_idx].to(DEVICE)
        X_test, Y_test = X[test_idx].to(DEVICE), Y[test_idx].to(DEVICE)
        print(f"训练: {len(X_train)} | 测试: {len(X_test)}")

        print("\n[4/5] 训练...")
        model, weights_history = train_model(X_train, Y_train, X_test, Y_test)

        print("\n[5/5] 评估...")
        model.eval()
        with torch.no_grad():
            pred_final, _ = model(X_test)

            physics_pred = None
            if BASELINE_EVAL:
                physics_pred = model.physics_baseline(X_test)
                evaluate(physics_pred, Y_test, "测试集 [物理]")

            final_metrics = evaluate(pred_final, Y_test, "测试集 [融合]")

            # ==================== 新增：公式输出与参数修改演示 ====================
            print("\n" + "="*60)
            print("当前训练得到的显式公式")
            print("="*60)
            model.print_formula('left')
            model.print_formula('right')
            model.export_formula(os.path.join(SAVE_DIR, "formula_default.json"))

            print("\n" + "="*60)
            print("【部署演示】根据反馈微调物理权重")
            print("="*60)
            print("假设实际落点反馈：左手位置预测偏大，右手速度预测偏小")
            old_pos_l = model.get_physics_weights('left')['pos']
            old_vel_r = model.get_physics_weights('right')['vel']
            model.set_physics_weight('left',  'pos', old_pos_l + 0.05)
            model.set_physics_weight('right', 'vel', old_vel_r - 0.03)
            model.print_formula('left')
            model.print_formula('right')
            model.export_formula(os.path.join(SAVE_DIR, "formula_finetuned.json"))
            # =====================================================================

            print("\n可视化...")
            plot_results(pred_final, Y_test, weights_history,
                       save_path=os.path.join(SAVE_DIR, "results.png"),
                       physics_pred=physics_pred)
            plot_3d_trajectory(pred_final, Y_test,
                              save_path=os.path.join(SAVE_DIR, "3d_landing.png"))

    print("\n" + "=" * 70)
    print("完成！输出:")
    print(f"  - {SAVE_DIR}/best_model*.pt")
    print(f"  - {SAVE_DIR}/formula_default.json   (原始公式)")
    print(f"  - {SAVE_DIR}/formula_finetuned.json (微调后公式)")
    print(f"  - {SAVE_DIR}/results_*.png")
    print(f"  - {SAVE_DIR}/3d_landing_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()