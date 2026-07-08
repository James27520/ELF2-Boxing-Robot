"""
config.py
V9版本：改进对齐 + 滑动窗口数据构建
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_FPS = 30.0

MAX_INPUT_T = 10
MIN_INPUT_T = 5
INPUT_T = MAX_INPUT_T

PRED_T = 1  # 只预测1帧终点
FPS = 30

HEIGHT_REAL = 1.68
SHOULDER_REAL = 0.41
HIP_WIDTH_REAL = 0.30
CAM_DISTANCE = 2.0

JOINTS = 17
NOSE = 0
LS, RS = 5, 6
LE, RE = 7, 8
LW, RW = 9, 10
HIP_L, HIP_R = 11, 12

TARGET_JOINTS = [LW, RW]
N_HANDS = 2
TARGET_JOINT = RW

TARGET_MAE = 0.01  # 1厘米精度
MAX_EPOCHS = 2000
LR = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 200

LOSS_WEIGHTS = {
    'final': 1.0,
    'physics': 0.2,
    'residual': 0.1,
    'weight_reg': 0.02,
}

YOLO_MODEL = "yolov8n-pose.pt"
DATA_DIRS = ["boxing_single_1", "boxing_single_2", "boxing_single_3"]
SAVE_DIR = "./outputs"

SCALE_SMOOTH_WINDOW = 5

HUMAN_MAX_ACCEL = 15.0
HUMAN_MAX_SPEED = 12.0
HUMAN_MAX_ANGULAR_VEL = 20.0

CROSS_VAL = True
N_FOLDS = 3

BASELINE_EVAL = True

MAX_PRED_RANGE = 2.0

# 迭代修正参数
ITERATIVE_REFINE = True
REFINE_WINDOW_SIZES = [10, 9, 8, 7, 6, 5]
REFINE_TOLERANCE = 0.01

# V9新增：动作检测参数
ACTION_DETECTION = True
WRIST_SPEED_THRESHOLD = 0.5  # 手腕速度阈值（像素/帧），超过此值认为动作开始
MIN_ACTION_FRAMES = 30  # 最小动作帧数（1秒）
LEADING_BUFFER = 10  # 动作开始前保留的帧数（用于预测上下文）
TRAILING_BUFFER = 10  # 动作结束后保留的帧数

DEBUG_SCALE = True
DEBUG_ACCEL = True
DEBUG_ALIGN = True

KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASUREMENT_NOISE = 1e-2

GRAVITY = 9.8