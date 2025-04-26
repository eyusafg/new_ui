import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def generate_contour(
    x_start, x_end, y_top, y_bottom, n_points=200, noise=2, outlier_prob=0.05,
    simulate_corner=False, corner_inward=True, corner_len_ratio=0.15, corner_depth=25,
    simulate_notch=False, notch_depth=30
):
    """
    随机生成一条近似矩形的轮廓（带轻微扰动和偶尔异常点）
    可选模拟斜角（往内/外斜）、缺口
    """
    x = np.linspace(x_start, x_end, n_points)
    y1 = y_top + np.random.randn(n_points) * noise + np.random.uniform(-1, 1, n_points) * noise
    y2 = y_bottom + np.random.randn(n_points) * noise + np.random.uniform(-1, 1, n_points) * noise

    # 斜角模拟
    if simulate_corner:
        corner_len = int(n_points * corner_len_ratio)
        if corner_inward:
            # 左上角往内斜
            y1[:corner_len] += -np.linspace(0, corner_depth, corner_len)
            # 右下角往内斜
            y2[-corner_len:] += np.linspace(0, -corner_depth, corner_len)
        else:
            # 左上角往外斜
            y1[:corner_len] += np.linspace(0, corner_depth, corner_len)
            # 右下角往外斜
            y2[-corner_len:] += np.linspace(0, corner_depth, corner_len)

    # 缺口模拟
    if simulate_notch:
        notch_start = n_points // 3
        notch_end = notch_start + n_points // 8
        y1[notch_start:notch_end] -= notch_depth  # 上边缘中间下凹
        y2[notch_start:notch_end] += notch_depth  # 下边缘中间上凸

    # 偶尔插入异常点
    for i in range(n_points):
        if np.random.rand() < outlier_prob:
            y1[i] += np.random.choice([20, -20, 30, -30])
        if np.random.rand() < outlier_prob:
            y2[i] += np.random.choice([20, -20, 30, -30])
    contour = np.vstack([np.column_stack([x, y1]), np.column_stack([x[::-1], y2[::-1]])])
    return contour

def find_max_min_y_for_x(contour, x_target, tol=1.0):
    """
    在轮廓中找到x最接近x_target的所有点，返回最大y和最小y
    """
    x_diff = np.abs(contour[:, 0] - x_target)
    indices = np.where(x_diff < tol)[0]
    if len(indices) == 0:
        # 没有找到，返回nan
        return np.nan, np.nan
    y_values = contour[indices, 1]
    return np.max(y_values), np.min(y_values)

def sample_edges(contour, x_start, x_end, step, direction='left2right'):
    """
    按步长采样x，返回每个x上的上下边缘点
    """
    if direction == 'left2right':
        x_samples = np.arange(x_start, x_end+1, step)
    else:
        x_samples = np.arange(x_start, x_end-1, -step)
    p1, p2 = [], []
    for x in x_samples:
        max_y, min_y = find_max_min_y_for_x(contour, x)
        p1.append([x, max_y])
        p2.append([x, min_y])
    return np.array(p1), np.array(p2), x_samples

def filter_by_height(p1, p2, threshold=2.5):
    """
    根据高度（上y-下y）剔除异常组，返回mask
    """
    heights = p1[:, 1] - p2[:, 1]
    median = np.median(heights)
    mad = np.median(np.abs(heights - median))
    # 使用中位数绝对偏差（MAD）判定异常
    mask = np.abs(heights - median) < threshold * mad
    return mask

def remove_edge_groups(p1, p2, edge_ratio=0.12):
    n = len(p1)
    edge_n = int(n * edge_ratio)
    mask = np.zeros(n, dtype=bool)
    mask[edge_n:n-edge_n] = True
    return mask

def detect_trend_change_mask(heights, threshold=8, min_edge=3):
    """
    检测高度序列的突变点，返回mask，突变点一侧的组全部剔除
    threshold: 差分突变阈值
    min_edge: 至少保留多少组不剔除
    """
    diffs = np.diff(heights)
    idx = np.argmax(np.abs(diffs))
    if np.abs(diffs[idx]) > threshold:
        # 以突变点为界，靠近边缘一侧全部剔除
        mask = np.ones_like(heights, dtype=bool)
        # 判断是左侧还是右侧突变
        if idx < len(heights) // 2:
            # 左侧突变，剔除左侧
            mask[:idx+1] = False
            mask[:min_edge] = False  # 保证最左侧一定剔除
        else:
            # 右侧突变，剔除右侧
            mask[idx+1:] = False
            mask[-min_edge:] = False  # 保证最右侧一定剔除
        return mask
    else:
        # 没有明显突变，全部保留
        return np.ones_like(heights, dtype=bool)

def sliding_window_trend_mask(heights, window=5, threshold=10):
    """
    滑动窗口趋势剔除，窗口均值偏离全局中位数较多的区间整体剔除
    """
    n = len(heights)
    global_median = np.median(heights)
    mask = np.ones(n, dtype=bool)
    for i in range(n - window + 1):
        local_mean = np.mean(heights[i:i+window])
        if np.abs(local_mean - global_median) > threshold:
            mask[i:i+window] = False
    return mask

# ====== 开关 ======
simulate_corner_head = True    # 前部是否有斜角
corner_inward_head = True      # 前部斜角是否往内
simulate_corner_tail = False   # 尾部是否有斜角
corner_inward_tail = True      # 尾部斜角是否往内
simulate_notch_head = False    # 前部是否有缺口
simulate_notch_tail = False    # 尾部是否有缺口

# 1. 生成前部和尾部轮廓
np.random.seed()  # 每次都不一样
contour_head = generate_contour(
    50, 250, 100, 200, n_points=200, noise=3, outlier_prob=0.08,
    simulate_corner=simulate_corner_head, corner_inward=corner_inward_head, corner_len_ratio=0.18, corner_depth=22,
    simulate_notch=simulate_notch_head, notch_depth=30
)
contour_tail = generate_contour(
    250, 50, 110, 210, n_points=200, noise=3, outlier_prob=0.08,
    simulate_corner=simulate_corner_tail, corner_inward=corner_inward_tail, corner_len_ratio=0.18, corner_depth=22,
    simulate_notch=simulate_notch_tail, notch_depth=30
)

# 2. 设定对齐点和终点
x_target_head = contour_head[:, 0].min()#（最左边）
x_border_head = contour_head[:, 0].max()#（最右边）
x_target_tail = contour_tail[:, 0].max()#（最右边）
x_border_tail = contour_tail[:, 0].min()#（最左边）
step = 7

# 3. 采样
p1_head, p2_head, x_samples_head = sample_edges(contour_head, x_target_head, x_border_head, step, 'left2right')
p1_tail, p2_tail, x_samples_tail = sample_edges(contour_tail, x_target_tail, x_border_tail, step, 'right2left')

# 4. 只保留数量一致的点（取最小长度）
min_len = min(len(p1_head), len(p1_tail))
p1_head, p2_head = p1_head[:min_len], p2_head[:min_len]
p1_tail, p2_tail = p1_tail[:min_len], p2_tail[:min_len]

## 第一种方法
# 5. 分别对前部和尾部做异常组剔除
# mask_head = filter_by_height(p1_head, p2_head, threshold=2.5)
# mask_tail = filter_by_height(p1_tail, p2_tail, threshold=2.5)
# # 新增：边缘区间剔除
# edge_mask = remove_edge_groups(p1_head, p2_head, edge_ratio=0.12)
# # 只保留前尾都为True的组
# final_mask = mask_head & mask_tail & edge_mask

## 第二种方法
# 5. 分别对前部和尾部做异常组剔除
# heights_head = p1_head[:, 1] - p2_head[:, 1]
# heights_tail = p1_tail[:, 1] - p2_tail[:, 1]
# mask_head = detect_trend_change_mask(heights_head, threshold=10, min_edge=3)
# mask_tail = detect_trend_change_mask(heights_tail, threshold=10, min_edge=3)
# # 只保留前尾都为True的组
# final_mask = mask_head & mask_tail

## 第三种方法
heights_head = p1_head[:, 1] - p2_head[:, 1]
heights_tail = p1_tail[:, 1] - p2_tail[:, 1]
mask_head = sliding_window_trend_mask(heights_head, window=5, threshold=10)
mask_tail = sliding_window_trend_mask(heights_tail, window=5, threshold=10)
final_mask = mask_head & mask_tail

p1_head_final = p1_head[final_mask]
p2_head_final = p2_head[final_mask]
p1_tail_final = p1_tail[final_mask]
p2_tail_final = p2_tail[final_mask]

# 6. 可视化
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title('前部采样点（剔除异常组后）')
plt.plot(contour_head[:, 0], contour_head[:, 1], 'gray', alpha=0.3, label='轮廓')
plt.plot(p1_head[:, 0], p1_head[:, 1], 'ro-', alpha=0.3, label='上边缘-全部')
plt.plot(p2_head[:, 0], p2_head[:, 1], 'bo-', alpha=0.3, label='下边缘-全部')
plt.plot(p1_head_final[:, 0], p1_head_final[:, 1], 'r*', label='上边缘-保留')
plt.plot(p2_head_final[:, 0], p2_head_final[:, 1], 'b*', label='下边缘-保留')
for i in range(len(p1_head_final)):
    plt.plot([p1_head_final[i, 0], p2_head_final[i, 0]], [p1_head_final[i, 1], p2_head_final[i, 1]], 'g-')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.title('尾部采样点（剔除异常组后）')
plt.plot(contour_tail[:, 0], contour_tail[:, 1], 'gray', alpha=0.3, label='轮廓')
plt.plot(p1_tail[:, 0], p1_tail[:, 1], 'ro-', alpha=0.3, label='上边缘-全部')
plt.plot(p2_tail[:, 0], p2_tail[:, 1], 'bo-', alpha=0.3, label='下边缘-全部')
plt.plot(p1_tail_final[:, 0], p1_tail_final[:, 1], 'r*', label='上边缘-保留')
plt.plot(p2_tail_final[:, 0], p2_tail_final[:, 1], 'b*', label='下边缘-保留')
for i in range(len(p1_tail_final)):
    plt.plot([p1_tail_final[i, 0], p2_tail_final[i, 0]], [p1_tail_final[i, 1], p2_tail_final[i, 1]], 'g-')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
