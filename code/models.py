"""
models.py
V8修复版 + 新增：物理权重可调接口、显式公式输出
原有代码完全保留，只在类末尾追加5个方法
"""

import torch
import torch.nn as nn
from config import *


class PINNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_pos_l = nn.Parameter(torch.tensor(0.8))
        self.w_vel_l = nn.Parameter(torch.tensor(1.2))
        self.w_acc_l = nn.Parameter(torch.tensor(0.8))
        self.w_ang_l = nn.Parameter(torch.tensor(0.3))
        self.w_omg_l = nn.Parameter(torch.tensor(0.1))
        self.w_imp_l = nn.Parameter(torch.tensor(0.2))
        self.w_trend_l = nn.Parameter(torch.tensor(0.3))
        self.w_grav_l = nn.Parameter(torch.tensor(0.1))

        self.w_pos_r = nn.Parameter(torch.tensor(0.8))
        self.w_vel_r = nn.Parameter(torch.tensor(1.2))
        self.w_acc_r = nn.Parameter(torch.tensor(0.8))
        self.w_ang_r = nn.Parameter(torch.tensor(0.3))
        self.w_omg_r = nn.Parameter(torch.tensor(0.1))
        self.w_imp_r = nn.Parameter(torch.tensor(0.2))
        self.w_trend_r = nn.Parameter(torch.tensor(0.3))
        self.w_grav_r = nn.Parameter(torch.tensor(0.1))

        feature_dim = 21
        total_features = feature_dim * JOINTS

        self.input_bn = nn.BatchNorm1d(total_features)

        self.lstm = nn.LSTM(total_features, 256, 3,
                           batch_first=True, dropout=0.3)

        self.residual_left = nn.Sequential(
            nn.Linear(256 + 3, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.residual_right = nn.Sequential(
            nn.Linear(256 + 3, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.output_scale = MAX_PRED_RANGE

        # V8修复：gravity_vec 改为 [1, 3]，便于和 [B, 3] 广播
        self.register_buffer('gravity_vec', 
                            torch.tensor([0.0, GRAVITY, 0.0]).view(1, 3))

    def physics_single(self, x, joint_idx, weights):
        p = x[:, -1, joint_idx, 0:3]      # [B, 3]
        v = x[:, -1, joint_idx, 3:6]      # [B, 3]
        a = x[:, -1, joint_idx, 6:9]      # [B, 3]

        if joint_idx == LW:
            ang = x[:, -1, joint_idx, [9, 11, 13, 14]].mean(dim=-1, keepdim=True)  # [B, 1]
            omg = x[:, -1, joint_idx, 17:18]   # [B, 1]
            imp = x[:, -1, joint_idx, 19:20]   # [B, 1]
        else:
            ang = x[:, -1, joint_idx, [10, 12, 15, 16]].mean(dim=-1, keepdim=True)  # [B, 1]
            omg = x[:, -1, joint_idx, 18:19]   # [B, 1]
            imp = x[:, -1, joint_idx, 20:21]   # [B, 1]

        T = PRED_T / FPS  # 1/30秒

        # trend: [B, 3]
        if x.shape[1] >= 3:
            trend = (x[:, -1, joint_idx, 0:3] - x[:, -3, joint_idx, 0:3]) / 2.0
        else:
            trend = x[:, -1, joint_idx, 3:6]

        # 所有项都保持 [B, 3] 维度
        physics = (
            weights['pos'] * p +                                    # [B, 3]
            weights['vel'] * v * T +                                # [B, 3]
            weights['acc'] * 0.5 * a * (T ** 2) +                   # [B, 3]
            weights['ang'] * ang * 0.3 +                            # [B, 1] 广播到 [B, 3]
            weights['omg'] * omg * 0.3 * T +                        # [B, 1] 广播到 [B, 3]
            weights['imp'] * imp * T / 0.5 +                      # [B, 1] 广播到 [B, 3]
            weights['trend'] * trend * T +                         # [B, 3]
            weights['grav'] * 0.5 * self.gravity_vec * (T ** 2)    # [1, 3] 广播到 [B, 3]
        )
        return physics  # [B, 3]

    def forward_single(self, x):
        B = x.shape[0]
        T = x.shape[1]

        w_l = {'pos': self.w_pos_l, 'vel': self.w_vel_l, 'acc': self.w_acc_l,
               'ang': self.w_ang_l, 'omg': self.w_omg_l, 'imp': self.w_imp_l, 
               'trend': self.w_trend_l, 'grav': self.w_grav_l}
        w_r = {'pos': self.w_pos_r, 'vel': self.w_vel_r, 'acc': self.w_acc_r,
               'ang': self.w_ang_r, 'omg': self.w_omg_r, 'imp': self.w_imp_r, 
               'trend': self.w_trend_r, 'grav': self.w_grav_r}

        physics_l = self.physics_single(x, LW, w_l)  # [B, 3]
        physics_r = self.physics_single(x, RW, w_r)  # [B, 3]

        physics_end = torch.stack([physics_l, physics_r], dim=1)  # [B, 2, 3]

        # LSTM 处理
        x_flat = x.reshape(B, T, -1)        # [B, T, 357]
        x_flat = x_flat.transpose(1, 2)      # [B, 357, T]
        x_flat = self.input_bn(x_flat)       # [B, 357, T]
        x_flat = x_flat.transpose(1, 2)      # [B, T, 357]

        h, _ = self.lstm(x_flat)
        h = h[:, -1]  # [B, 256]

        # 残差网络：h [B, 256] + physics_l [B, 3] -> [B, 259]
        residual_l = self.residual_left(torch.cat([h, physics_l], dim=-1))  # [B, 3]
        residual_l = torch.tanh(residual_l / self.output_scale) * self.output_scale

        residual_r = self.residual_right(torch.cat([h, physics_r], dim=-1))  # [B, 3]
        residual_r = torch.tanh(residual_r / self.output_scale) * self.output_scale

        residual = torch.stack([residual_l, residual_r], dim=1)  # [B, 2, 3]

        final_pred = physics_end + residual  # [B, 2, 3]
        final_pred = torch.tanh(final_pred / self.output_scale) * self.output_scale

        return final_pred, {
            'physics_traj': physics_end,
            'residual': residual,
            'weights': {
                'left': {k: v.item() for k, v in w_l.items()},
                'right': {k: v.item() for k, v in w_r.items()}
            }
        }

    def forward(self, x, valid_len=None):
        if self.training or valid_len is not None:
            return self.forward_single(x)
        else:
            return self.iterative_predict(x)

    def iterative_predict(self, x):
        """迭代修正预测"""
        B = x.shape[0]

        predictions = []
        confidences = []

        for window_size in REFINE_WINDOW_SIZES:
            if window_size > x.shape[1]:
                continue
            x_window = x[:, -window_size:]
            pred, model_out = self.forward_single(x_window)
            predictions.append(pred)
            confidences.append(window_size / 10.0)

        total_conf = sum(confidences)
        weights = [c / total_conf for c in confidences]

        fused_pred = sum(p * w for p, w in zip(predictions, weights))

        _, final_out = self.forward_single(x)

        return fused_pred, final_out

    def physics_baseline(self, x):
        B = x.shape[0]

        w_l = {'pos': self.w_pos_l, 'vel': self.w_vel_l, 'acc': self.w_acc_l,
               'ang': self.w_ang_l, 'omg': self.w_omg_l, 'imp': self.w_imp_l, 
               'trend': self.w_trend_l, 'grav': self.w_grav_l}
        w_r = {'pos': self.w_pos_r, 'vel': self.w_vel_r, 'acc': self.w_acc_r,
               'ang': self.w_ang_r, 'omg': self.w_omg_r, 'imp': self.w_imp_r, 
               'trend': self.w_trend_r, 'grav': self.w_grav_r}

        physics_l = self.physics_single(x, LW, w_l)  # [B, 3]
        physics_r = self.physics_single(x, RW, w_r)  # [B, 3]

        physics_end = torch.stack([physics_l, physics_r], dim=1)  # [B, 2, 3]

        return physics_end

    # ==================== 以下为新增方法 ====================

    def set_physics_weight(self, hand, name, value):
        """
        部署时修改单个物理权重。
        hand: 'left' 或 'right'
        name: 'pos'/'vel'/'acc'/'ang'/'omg'/'imp'/'trend'/'grav'
        value: 新的浮点数值
        """
        suffix = 'l' if hand == 'left' else 'r'
        param_name = f'w_{name}_{suffix}'
        if not hasattr(self, param_name):
            raise ValueError(f"未知参数: {param_name}")
        param = getattr(self, param_name)
        with torch.no_grad():
            param.copy_(torch.tensor(value, dtype=param.dtype, device=param.device))

    def set_physics_weights(self, hand, weights_dict):
        """批量修改物理权重，如 {'pos': 1.1, 'vel': 0.95}"""
        for name, value in weights_dict.items():
            self.set_physics_weight(hand, name, value)

    def get_physics_weights(self, hand='left'):
        """获取当前8个物理权重的数值字典"""
        suffix = 'l' if hand == 'left' else 'r'
        names = ['pos', 'vel', 'acc', 'ang', 'omg', 'imp', 'trend', 'grav']
        return {name: getattr(self, f'w_{name}_{suffix}').item() for name in names}

    def print_formula(self, hand='left'):
        """打印当前权重下的显式预测公式"""
        w = self.get_physics_weights(hand)
        dt = PRED_T / FPS
        print(f"\n{'='*60}")
        print(f"{'左手' if hand=='left' else '右手'} 落点预测公式")
        print(f"  dt = {dt:.4f}s (1/{FPS}秒)")
        print(f"{'='*60}")
        terms = [
            (w['pos'],    "P",                  "当前位置"),
            (w['vel'],    f"V × {dt:.4f}",      "速度 × dt"),
            (w['acc'],    f"0.5 × A × {dt**2:.6f}", "0.5 × 加速度 × dt²"),
            (w['ang'],    "Ang × 0.3",          "平均角度 × 0.3"),
            (w['omg'],    f"Omg × 0.3 × {dt:.4f}", "角速度 × 0.3 × dt"),
            (w['imp'],    f"Imp × {dt:.4f} / 0.5", "冲量 × dt / 0.5"),
            (w['trend'],  f"Trend × {dt:.4f}",  "趋势 × dt"),
            (w['grav'],   f"0.5 × [0,{GRAVITY},0] × {dt**2:.6f}", "重力项"),
        ]
        for i, (coeff, expr, desc) in enumerate(terms):
            print(f"  项{i+1}: {coeff:+.4f} × {expr:22s}  # {desc}")
        print(f"  + LSTM残差网络校正项")
        print(f"{'='*60}")

    def export_formula(self, output_path="formula.json"):
        """导出物理权重为JSON，供C++部署或人工查看"""
        import json
        dt = PRED_T / FPS
        formula = {
            'meta': {'dt': dt, 'fps': FPS, 'pred_t': PRED_T, 'gravity': GRAVITY},
            'left':  self.get_physics_weights('left'),
            'right': self.get_physics_weights('right'),
        }
        with open(output_path, 'w') as f:
            json.dump(formula, f, indent=2)
        print(f"✓ 公式系数已导出: {output_path}")
        return formula