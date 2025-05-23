import cv2
import numpy as np
# from scipy.stats import gaussian_kde
# from scipy import stats
# from collections import Counter
from .get_max_contour import get_max_contour
import pandas as pd

def find_nearest_point(point, points):
    distances = np.linalg.norm(points - point, axis=1)
    nearest_index = np.argmin(distances)
    return points[nearest_index]


def find_max_min_y_for_x(contour, x_target):
    """
    在给定的轮廓中找到对应x坐标的最大和最小的y值。

    :param contour: 轮廓，形状为 (n_points, 1, 2)，其中每个点是一个(x, y)坐标
    :param x_target: 目标x坐标
    :return: 包含最大和最小y值的数组，格式为 [max_y, min_y]
    """

    # 计算每个点的x坐标与x_target的差值
    x_diff = np.abs(contour[:, 0] - x_target)

    # 找到x_diff中最小的差值对应的索引
    nearest_indices = np.where(x_diff == np.min(x_diff))[0]

    # 筛选出最近x坐标对应的y值
    y_values = contour[nearest_indices, 1]
    # x_values = contour[nearest_indices, 0]

    # # 筛选出x坐标匹配的点
    # y_values = contour_array[contour_array[:, 0] == x_target, 1]

    # if y_values.size == 0:
    #     return None  # 如果没有找到匹配的点，返回None

    # 找出这些点中的y坐标的最大值和最小值
    max_y = np.max(y_values)
    min_y = np.min(y_values)

    return max_y, min_y

def get_kde_x(mask, approx_points, points, name, visualize=False):
    '''
    只需要获取上到下的x的边缘坐标
    '''
    
    p1 = points[0]
    p2 = points[1]

    # # Find indices of p1 and p2 in the contour
    idx1 = find_nearest_point_index(approx_points, p1)
    idx2 = find_nearest_point_index(approx_points, p2)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    # n_points = len(approx_points)
    pb2 = np.concatenate([approx_points[idx2+1:], approx_points[:idx1]])
    pb1 = approx_points[idx1:idx2+1]

    if len(pb1) <= len(pb2):
        points_between = pb1
    else:
        points_between = pb2
    
    # df = pd.DataFrame(points_between, columns=['x', 'y'])
    # df.to_csv(f'{name}_points_between.csv', index=False)

    # 按照y的大小进行排序
    sorted_indices = np.argsort(points_between[:, 1])
    points_between = points_between[sorted_indices]
    # print('points_between', points_between)
    # df = pd.DataFrame(points_between, columns=['x', 'y'])
    # df.to_csv(f'{name}_points_between_sorted.csv', index=False)
    # 取线段的x值
    x_coords = points_between[:, 0]
    # print('x_coords dtype', x_coords.dtype)
    # print("X coordinates:", x_coords) 

    ############ 重复值判断平直段 #### 
    # p1_y_counts = Counter(p1_y_value)
    # most_common_y1 = p1_y_counts.most_common()
    # p2_y_counts = Counter(p2_y_value)
    # most_common_y2 = p2_y_counts.most_common()
    
    # num_most_common = 4
    # most_common_y1 = [y for y, count in most_common_y1[:num_most_common]]
    # most_common_y2 = [y for y, count in most_common_y2[:num_most_common]]
    
    # if loc == 'head':
    #     y = most_common_y1[0] 
        # cv2.line(mask_bgr, (int(550), int(1)),(int(550), int(1000)), (0, 255, 0), 2) #  roi边缘位置
    # else:
    #     y = most_common_y2[0]

    # for point in p1_extended:
    #     cv2.circle(mask_bgr, (int(point[0]), int(most_common_y1[0])), 2, (255, 0, 0), -1)

    # for point in p2_extended:
    #     cv2.circle(mask_bgr, (int(point[0]), int(most_common_y2[0])), 2, (0, 0, 255), -1)
    ############ 重复值判断平直段 #### 
  

    #########################计算质心#######################################
    # 过滤异常值
    # z_scores = np.abs(stats.zscore(x_coords))
    # filtered_x = x_coords[z_scores < 2]  # 过滤2σ外的异常点
    # x_centroid = np.mean(filtered_x)  # 计算均值当质心

    # # 可视化验证
    # if visualize:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10,5))
        
    #     # 原始轮廓可视化
    #     plt.subplot(121)
    #     plt.plot(points_between[:,0], points_between[:,1], 'b-', alpha=0.3)
    #     plt.plot(p1[0], p1[1], 'ro', markersize=10)
    #     plt.plot(p2[0], p2[1], 'ro', markersize=10)
    #     plt.title('原始轮廓及角点位置')
    #     plt.axis('equal')
        
    #     # 处理结果可视化
    #     plt.subplot(122)
    #     plt.plot(points_between[:,0], points_between[:,1], 'g-', alpha=0.5, label='线段轮廓')
    #     plt.axvline(x_centroid, color='r', linestyle='--', label='X质心')
    #     plt.scatter(x_centroid, np.mean(points_between[:,1]), 
    #                c='r', s=100, marker='x', linewidth=2)
    #     plt.legend()
    #     plt.title(f'X质心位置：{x_centroid:.2f}')
    #     plt.axis('equal')
    #     plt.show()
    #########################计算质心#######################################

    #########################计算均值和标准差#######################################
    # 计算均值和标准差  和聚类效果基本一致
    # mean_x = np.mean(x_coords)
    # std_dev_x = np.std(x_coords)
    # print(f'mean_x: {mean_x}, std_dev_x: {std_dev_x}')

    # threshold = 0.5  # 3 standard deviations
    # outliers = (abs(x_coords - mean_x) > threshold * std_dev_x)

    # filtered_x_coords = x_coords[~outliers]
    # print("Filtered X coordinates:", filtered_x_coords)
    # mean_x_inliers = np.mean(filtered_x_coords)
    # cv2.line(mask_bgr, (int(mean_x_inliers), int(point[1])),(int(mean_x_inliers), int(int(p1[1]))), (255, 0, 255), 2)
    #########################计算均值和标准差#######################################

    # 滤波
    # window_size = 5  # 滤波窗口大小
    # filtered_x_coords_medfilt = medfilt(x_coords, kernel_size=window_size)
    # print("Median filtered X coordinates:", filtered_x_coords_medfilt)

    #########################聚类#######################################
    # db = DBSCAN(eps=0.3, min_samples=10).fit(points_between)
    # labels = db.labels_

    # # 获取最大簇的索引
    # max_cluster_label = max(set(labels), key=list(labels).count)
    # inliers = labels == max_cluster_label
    # print("Inlier X coordinates from DBSCAN:", points_between[inliers, 0])
    # mean_x_inliers = np.mean(points_between[inliers, 0])
    # cv2.line(mask_bgr, (int(mean_x_inliers), int(point[1])),(int(mean_x_inliers), int(int(p1[1]))), (255, 0, 255), 2)
    #########################聚类#######################################

    #########################最小二乘#######################################
    # A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    # m, c = np.linalg.lstsq(A, points_between[:, 1], rcond=None)[0]  # m is slope, c is intercept

    # # Calculate distances from each point to the line
    # distances = abs(m * x_coords - points_between[:, 1] + c) / np.sqrt(m**2 + 1)

    # # Define a threshold for what you consider an outlier based on distance
    # distance_threshold = 2  # Example value
    # inliers_line_fit = distances < distance_threshold
    # print("Inlier X coordinates from line fit:", x_coords[inliers_line_fit])
    # mean_x_inliers = np.mean(x_coords[inliers_line_fit])
    # cv2.circle(mask_bgr, (int(mean_x_inliers), int(point[1])), 10, (255, 0, 255), -1)
    #########################最小二乘#######################################

    #########################基于密度/直方图#######################################
    # 基于密度/直方图  更贴近实际情况       ！！！！！！！！！！！！！！！！！！！！！
    # density = gaussian_kde(x_coords)
    # xs = np.linspace(min(x_coords), max(x_coords), 20)
    # density_estimation = density(xs)
    # max_density_index = np.argmax(density_estimation)
    # max_density_x = xs[max_density_index]
    # print(f"Maximum density is at x={max_density_x}")

    # cv2.circle(mask_bgr, (int(max_density_x), int(point[1])), 10, (255, 0, 255), -1)
    # cv2.line(mask_bgr, (int(max_density_x), int(point[1])),(int(max_density_x), int(int(p1[1]))), (255, 0, 255), 1)

    # plt.plot(xs, density_estimation)
    # plt.fill_between(xs, density_estimation, alpha=0.5)
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Density')
    # plt.title('Kernel Density Estimation of X Coordinates')
    # plt.show()
    #########################基于密度/直方图#######################################

    if visualize:
        # 绘制轮廓点
        # for point in approx_points:
        #     cv2.circle(mask_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255 ), -1)

        # 绘制 p1 和 p2
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(mask_bgr, tuple(p1.astype(int)), 10, (0,255,0), -1)
        cv2.circle(mask_bgr, tuple(p2.astype(int)), 10, (0,255,0), -1)
        
        # 绘制 p1 和 p2 之间的轮廓点
        for point in points_between:                 
            cv2.circle(mask_bgr, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

        # for i in approx:
        #     cv2.circle(mask_bgr, (int(i[0][0]), int(i[0][1])), 5, (0, 0, 255), -1)
        # cv2.circle(mask_bgr, (int(p1[0]), int(p1[1])), 5, (0, 255, 0), -1)
        # cv2.circle(mask_bgr, (int(p2[0]), int(p2[1])), 5, (0, 255, 0), -1)

        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", mask_bgr)
        cv2.waitKey(0)
  
    return x_coords, points_between
              
def get_kde_y(mask, x_target, approx_points, w, loc='head', num=None, visualize=True):
    """
    获取y坐标点集
    :param mask: 分割推理图像
    :param x_target: 优化过后的x坐标
    :param approx_points: 轮廓点集
    :param loc: 属于前部还是尾部
    :param length: 获取轮廓坐标点的长度
    :param num: 前部优化过后的y的数量， 尾部需要与其保持一致
    :param visualize: 可视化
    :return: 坐标点集 后续使用y需要 p1[:, 1], p2[:, 1]
    """

    '''
    首先需要通过上述 get_kde 获取到x的坐标点集以及轮廓点的坐标 然后直接输入get_kde_ 则可以省去重新输入mask计算轮廓的步骤
    '''
    
    # w = x_border - x_target
    w_range_num = w // 10

    p1, p2 = [], []
    for _ in range(w_range_num):
        max_y, min_y = find_max_min_y_for_x(approx_points, x_target)
        p1.append(np.array([x_target, max_y]))
        p2.append(np.array([x_target, min_y])) 
        if loc == 'head':
            x_target += 10
        else:
            x_target -= 10

    p1 = np.array(p1)
    p2 = np.array(p2)
    p1_y_value = p1[:, 1]  # 取y
    p2_y_value = p2[:, 1]  # 取y
    # print('p1_y_value', p1_y_value)
    # print('p2_y_value', p2_y_value)

    '''
    主要是去除异常值
    '''
    offset_y = p1_y_value - p2_y_value
    print(loc, 'offset_y', offset_y)
    max_offset_w = np.max(np.abs(offset_y))
    max_deviation = 40
    mask_ = np.abs(offset_y) >= (max_offset_w - max_deviation)
    # filtered_p1_y_value = p1_y_value[mask_]
    # filtered_p2_y_value = p2_y_value[mask_]
    # filtered_p1_xy_value = p1[0:num_head]
    # filtered_p2_xy_value = p2[0:num_head]
    p1 = p1[mask_]
    p2 = p2[mask_] 
    # print('p1', p1)
    # print('p2', p2)
    num_head = len(p1)
    if num is not None:
        p1 = p1[-num:]
        p2 = p2[-num:]


    #########  下面是直接按照轮廓顺序来取坐标#############
    # # Find indices of p1 and p2 in the contour
    # idx1 = find_nearest_point_index(approx_points, p1)
    # idx2 = find_nearest_point_index(approx_points, p2)

    # # Ensure idx1 < idx2
    # if idx1 > idx2:
    #     direction_idx1 = 'forward'
    #     direction_idx2 = 'backward'
    #     # idx1, idx2 = idx2, idx1
    # else:
    #     direction_idx1 = 'backward'
    #     direction_idx2 = 'forward'


    # # Calculate points along the outer side of the contour (pb2)
    # n_points = len(approx_points)
    # pb2 = np.concatenate([approx_points[idx2+1:], approx_points[:idx1]])

    # # Get 150 points from p1 and p2 along the outer contour
    # # def get_points_around(index, length, direction='backward'):
    # #     indices = []
    # #     current = index
    # #     for _ in range(length):
    # #         if direction == 'backward':
    # #             current = (current - 1) % n_points
    # #         else:
    # #             current = (current + 1) % n_points
    # #         indices.append(current)
    # #     return approx_points[indices]

    # def get_points_around(index, length, direction='backward'):
    #     indices = []
    #     current = index
    #     for _ in range(length):
    #         if direction == 'backward':
    #             current = (current - 1) % n_points  # 逆时针移动
    #         else:
    #             current = (current + 1) % n_points  # 顺时针移动
    #         indices.append(current)
    #     return approx_points[indices]

    # # p1: move backward along the outer contour
    # p1_extended = get_points_around(idx1, length, direction_idx1)
    # # p2: move forward along the outer contour
    # p2_extended = get_points_around(idx2, length, direction_idx2)

    # # Sort based on location
    # # if loc == 'head':
    # #     p1_extended = sorted(p1_extended, key=lambda x: -x[1])[:length]
    # #     p2_extended = sorted(p2_extended, key=lambda x: x[1])[:length]
    # # else:
    # #     p1_extended = sorted(p1_extended, key=lambda x: x[1])[:length]
    # #     p2_extended = sorted(p2_extended, key=lambda x: -x[1])[:length]

    # if loc == 'head':
    #     # p1沿逆时针方向取的原始点序列（直接截取）
    #     p1_extended_ = p1_extended[:length]
    #     print('p1_extended_:', p1_extended_.shape)
    #     # p2沿顺时针方向取的原始点序列（直接截取）
    #     p2_extended_ = p2_extended[:length]
    # else:
    #     # 尾部处理同理
    #     p1_extended_ = p2_extended[:length]
    #     p2_extended_ = p1_extended[:length]

    # # 计算步长
    # # step = len(p1_extended_) // 50
    # # # 生成等分的索引
    # # indices = np.arange(0, len(p1_extended_), step)
    # # p1_extended_ = p1_extended_[indices]
    # # p2_extended_ = p2_extended_[indices]

    # p1_y_value = p1_extended_[:, 0]  # 取x
    # p2_y_value = p2_extended_[:, 0]  # 取x

    # offset_x = p2_y_value - p1_y_value
    # max_deviation = 5
    # mask_ = np.abs(offset_x) <= max_deviation
    # filtered_p1_y_value = p1_y_value[mask_]
    # filtered_p2_y_value = p2_y_value[mask_]
    # num_head = len(filtered_p1_y_value)
    # filtered_p1_xy_value = p1_extended_[0:num_head]
    # filtered_p2_xy_value = p2_extended_[0:num_head]

    # def filter_points(p1_list, p2_list):

    #     # 第一层过滤：y差值对比最大差值
    #     # 计算所有y差值并找到最大值
    #     y_diffs = [abs(p1[1]-p2[1]) for p1, p2 in zip(p1_list, p2_list)]
    #     max_y_diff = max(y_diffs)
        
    #     # 保留差值在[max-10, max]范围内的点对
    #     filtered_p1, filtered_p2 = [], []
    #     for p1, p2 in zip(p1_list, p2_list):
    #         current_diff = abs(p1[1]-p2[1])
    #         if abs(max_y_diff - current_diff) <= 50:
    #             filtered_p1.append(p1)
    #             filtered_p2.append(p2)
        
    #     # 第二层过滤：x差值不超过3
    #     final_p1, final_p2 = [], []
    #     for p1, p2 in zip(filtered_p1, filtered_p2):
    #         if abs(p1[0] - p2[0]) <= 3:
    #             final_p1.append(p1)
    #             final_p2.append(p2)
        
    #     return final_p1, final_p2
    
    # def filter_points_(p1_list, p2_list):
    #     if len(p1_list) < 10 or len(p2_list) < 10:
    #         return p1_list, p2_list

    #     # 功能1：计算点集的平均移动趋势 ------------------------------------------------
    #     def calc_trend_segment(points, start_idx, end_idx):
    #         """计算指定区间点的平均移动方向"""
    #         dx = points[end_idx][0] - points[start_idx][0]
    #         dy = points[end_idx][1] - points[start_idx][1]
    #         return dx, dy

    #     # 功能2：获取基准趋势 ------------------------------------------------------
    #     # 取前1/3长度的点作为基准趋势段（至少5个点）
    #     # sample_size = max(5, len(p1_list)//3)
    #     sample_size = 5
    #     p1_base_trend = calc_trend_segment(p1_list, 0, sample_size-1)
    #     p2_base_trend = calc_trend_segment(p2_list, 0, sample_size-1)

    #     # 功能3：动态趋势过滤 ------------------------------------------------------
    #     window_size = 10  # 增大窗口以提高稳定性
    #     trend_threshold = 0.8  # 趋势相似度阈值（0-1）
    #     min_keep_points = 20  # 最小保留点数
        
    #     valid_mask = [False]*len(p1_list)
        
    #     # 滑动窗口检测（带重叠）
    #     for i in range(0, len(p1_list)-window_size+1, window_size//2):
    #         # 获取当前窗口
    #         p1_win = p1_list[i:i+window_size]
    #         p2_win = p2_list[i:i+window_size]
            
    #         # 计算窗口趋势
    #         p1_win_trend = calc_trend_segment(p1_win, 0, -1)
    #         p2_win_trend = calc_trend_segment(p2_win, 0, -1)
            
    #         # 计算趋势相似度（余弦相似度）
    #         def cosine_similarity(vec1, vec2):
    #             dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    #             norm1 = (vec1[0]**2 + vec1[1]**2)**0.5
    #             norm2 = (vec2[0]**2 + vec2[1]**2)**0.5
    #             return dot/(norm1*norm2 + 1e-6)
            
    #         # 双重趋势检查
    #         sim_to_p1base = cosine_similarity(p1_win_trend, p1_base_trend)
    #         sim_to_p2base = cosine_similarity(p2_win_trend, p2_base_trend)
            
    #         # 当两个趋势同时符合要求时才保留
    #         if sim_to_p1base > trend_threshold and sim_to_p2base > trend_threshold:
    #             valid_mask[i:i+window_size] = [True]*window_size

    #     # 功能4：结果后处理 --------------------------------------------------------
    #     # 连接连续的有效区间（至少保留min_keep_points个点）
    #     final_p1, final_p2 = [], []
    #     current_segment = []
        
    #     for i, valid in enumerate(valid_mask):
    #         if valid:
    #             current_segment.append(i)
    #         else:
    #             if len(current_segment) >= min_keep_points:
    #                 # 保存有效段
    #                 final_p1.extend(p1_list[current_segment[0]:current_segment[-1]+1])
    #                 final_p2.extend(p2_list[current_segment[0]:current_segment[-1]+1])
    #             current_segment = []
        
    #     # 处理最后一段
    #     if len(current_segment) >= min_keep_points:
    #         final_p1.extend(p1_list[current_segment[0]:current_segment[-1]+1])
    #         final_p2.extend(p2_list[current_segment[0]:current_segment[-1]+1])

    #     return final_p1, final_p2

    # # 过滤异常值
    # # p1_extended, p2_extended = filter_points_(p1_extended_, p2_extended_)

    # Visualization
    if visualize:
        cv2.namedWindow(loc, cv2.WINDOW_NORMAL)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(mask_bgr, tuple(p1.astype(int)), 10, (0,255,0), -1)
        # cv2.circle(mask_bgr, tuple(p2.astype(int)), 10, (0,255,0), -1)
        for pt in p1:
            cv2.circle(mask_bgr, tuple(pt.astype(int)), 2, (255,0,0), -1)
        for pt in p2:
            cv2.circle(mask_bgr, tuple(pt.astype(int)), 2, (0,0,255), -1)

        for i in range(min(len(p1), len(p2))):
            # 画连接线
            cv2.line(mask_bgr, 
                    tuple(p1[i].astype(int)),
                    tuple(p2[i].astype(int)),
                    (0,255,0), 1)
        cv2.imshow(loc, mask_bgr)
        cv2.waitKey(0)
    '''
    p1为上边缘
    p2为下边缘
    '''

    return p1, p2, num_head

def find_nearest_point_index(points, target):
    distances = np.linalg.norm(points - target, axis=1)
    return np.argmin(distances)

def get_y(mask, contour_head, x_target_head, x_border_head, pred, contour_tail,x_target_tail,x_border_tail, step, visualize):
    p1_head, p2_head, x_samples_head = sample_edges(contour_head, x_target_head, x_border_head, step, 'head')
    p1_tail, p2_tail, x_samples_tail = sample_edges(contour_tail, x_target_tail, x_border_tail, step, 'tail')

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
    '''
    需要加一个报错信息， 如果没有检测到有效点， 取出现比较多的值
    '''
    if len(p1_head) < 3 or len(p2_head) < 3:
        return None, None, None, None
    if len(p1_tail) < 3 or len(p2_tail) < 3:
        return None, None, None, None
    
    heights_head = p1_head[:, 1] - p2_head[:, 1]
    heights_tail = p1_tail[:, 1] - p2_tail[:, 1]

    mask_head = detect_trend_change_mask(heights_head, threshold=10, min_edge=3)
    mask_tail = detect_trend_change_mask(heights_tail, threshold=10, min_edge=3)

    # 只保留前尾都为True的组
    final_mask = mask_head & mask_tail

    ## 第三种方法
    # heights_head = p1_head[:, 1] - p2_head[:, 1]
    # heights_tail = p1_tail[:, 1] - p2_tail[:, 1]
    # mask_head = sliding_window_trend_mask(heights_head, window=5, threshold=10)
    # mask_tail = sliding_window_trend_mask(heights_tail, window=5, threshold=10)
    # final_mask = mask_head & mask_tail

    p1_head_final = p1_head[final_mask]
    p2_head_final = p2_head[final_mask]
    p1_tail_final = p1_tail[final_mask]
    p2_tail_final = p2_tail[final_mask]

    # if len(p1_head_final) > 6:
    #     p1_head_final = p1_head_final[3:]
    #     p2_head_final = p2_head_final[3:]
    #     p1_tail_final = p1_tail_final[3:]
    #     p2_tail_final = p2_tail_final[3:]
        
    if visualize:
        cv2.namedWindow('head Points', cv2.WINDOW_NORMAL)
        cv2.namedWindow('tail Points', cv2.WINDOW_NORMAL)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        # cv2.circle(mask_bgr, tuple(p1.astype(int)), 10, (0,255,0), -1)
        # cv2.circle(mask_bgr, tuple(p2.astype(int)), 10, (0,255,0), -1)
        for pt in p1_head_final:
            cv2.circle(mask_bgr, tuple(pt.astype(int)), 2, (255,0,0), -1)
        for pt in p2_head_final:
            cv2.circle(mask_bgr, tuple(pt.astype(int)), 2, (0,0,255), -1)

        for i in range(min(len(p1_head_final), len(p2_head_final))):
            # 画连接线
            cv2.line(mask_bgr, 
                    tuple(p1_head_final[i].astype(int)),
                    tuple(p2_head_final[i].astype(int)),
                    (0,255,0), 1)
        cv2.line(mask_bgr, 
                    tuple([x_target_head.astype(int), 0]),
                    tuple([x_target_head.astype(int), 2000]),
                    (0,0,255), 1)
        cv2.imshow('head Points', mask_bgr)
        for pt in p1_tail_final:
            cv2.circle(pred_bgr, tuple(pt.astype(int)), 2, (255,0,0), -1)
        for pt in p2_tail_final:
            cv2.circle(pred_bgr, tuple(pt.astype(int)), 2, (0,0,255), -1)

        for i in range(min(len(p1_tail_final), len(p2_tail_final))):
            # 画连接线
            cv2.line(pred_bgr, 
                    tuple(p1_tail_final[i].astype(int)),
                    tuple(p2_tail_final[i].astype(int)),
                    (0,255,0), 1)
        cv2.line(pred_bgr, 
                    tuple([x_target_tail.astype(int), 0]),
                    tuple([x_target_tail.astype(int), 2000]),
                    (0,0,255), 1)
        cv2.imshow('tail Points', pred_bgr)
        cv2.waitKey(0)
    return p1_head_final, p2_head_final, p1_tail_final, p2_tail_final

def sample_edges(contour, x_start, x_end, step, direction='head'):
    """
    按步长采样x，返回每个x上的上下边缘点
    """
    if direction == 'head':
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

def get_min_bounding_rect_for_range(approx_points, x_target, x_border):
    # 计算x范围的右边界
    # x_border = x_target + w
    
    # 提取在x_target到x_border范围内的点
    selected_points = []
    
    for point in approx_points:
        x, y = point[0]  # assuming point format is (x, y)
        if x_target <= x <= x_border:
            selected_points.append([x, y])
    
    # 如果选中的点数大于等于4，才可以计算最小外接矩形
    if len(selected_points) >= 4:
        # 将点转换为numpy数组
        selected_points = np.array(selected_points)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(selected_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return rect
    else:
        return None
        
def detect_trend_change(points, angle_threshold=45, min_points=10):
    """
    检测点集的方向突变点
    :param points: 输入点集 (N, 2)
    :param angle_threshold: 角度变化阈值（度）
    :param min_points: 最小有效点数量（少于该数量直接返回无突变）
    :return: 突变点索引（若无突变返回-1）
    """
    if len(points) < min_points:
        return -1  # 点太少，认为无突变
    
    angles = []
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        if dx == 0:
            angle = 90 if dy > 0 else -90
        else:
            angle = np.degrees(np.arctan2(dy, dx))
        angles.append(angle)
    
    # 计算连续角度差
    angle_diffs = np.abs(np.diff(angles))
    # 找到超过阈值的突变点
    for idx in range(len(angle_diffs)):
        if angle_diffs[idx] > angle_threshold:
            return idx + 1  # 返回突变点的索引（相对于points）
    return -1

def infer_frame(image, sess, input_name, output_names):
    img = np.asarray([image]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})
    return detection

def infer_segm(image, sess, input_name, output_names):
    frame_se = cv2.resize(image, (384, 384))
    frame_se = np.ascontiguousarray(frame_se[:, :, ::-1].transpose(2, 0, 1))
    img = np.asarray([frame_se]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})[0]
    return detection

def load_model(model_path, delay=False):
    if delay: return None # 延迟加载分割模型
    
    so = rt.SessionOptions()
    so.intra_op_num_threads = 2
    # return rt.InferenceSession(model_path, so)
    sess = rt.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    return sess, input_name, output_names

if __name__ == "__main__":
    import onnxruntime as rt
    import os

    model_path = r'models\thor_keypoint_0515_00039_mod_data.onnx'
    segm_model_path = r'models\thor_segm_20250424_model_69.onnx'
    sess, input_name, output_name = load_model(model_path)
    segm_sess, segm_input_name, segm_output_name = load_model(segm_model_path, delay=False)

    # im_path = r'Data\thor_keypoint_data\Data\thor_kepoint\Data\2025_04_09_11_49_47_start.png'
    im_path = r'C:\Data\thor_kepoint\error_detect\20250521'
    files = os.listdir(im_path)

    for index, file in enumerate(files):
        print('file', file)
            
        # if file != '2025_05_15_21_40_49_start.png':
        #     continue 
        im = cv2.imread(im_path + '//' + file)
        # cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        # cv2.imshow('im', im)
        im_ = im[:, :, ::-1].transpose(2, 0, 1)
        de = infer_frame(im_, sess, input_name, output_name)
        se = infer_segm(im, segm_sess, segm_input_name, segm_output_name)
        pred = np.where(se.squeeze() > 0, 255, 0).astype(np.uint8)

        pts = de[0][0].astype(np.float64)
        # pts[:, 0] *= 1.1
        if 'start' in file:
            pts[:, 1] *= 1.57
            pred = cv2.resize(pred, (800, 2050))
        else:
            pts[:, 0] *= 1.875
            pts[:, 1] *= 1.92
            pred = cv2.resize(pred, (1500, 2500))

        max_contour = get_max_contour(pred)

        print('pts', pts)
        print(abs(abs(pts[0][0]) - abs(pts[1][0])))
        if abs(abs(pts[0][0]) - abs(pts[1][0])) >= 100:
            print('布料翘起')
        pts_y = pts[pts[:, 1].argsort()] 
        if 'start' in file:
            head_x_coords = get_kde_x(pred, max_contour, pts, 'head',True) # 返回边缘x坐标列表， 用于与尾部计算偏移
            p1_head, p2_head, x_samples_head = sample_edges(max_contour, 500, 603, 10, 'head')
        else:
            p1_head, p2_head, x_samples_head = sample_edges(max_contour, 560, 465 , 10, 'tail')

        heights_head = p1_head[:, 1] - p2_head[:, 1]
        mask_head = detect_trend_change_mask(heights_head, threshold=10, min_edge=3)
        p1_head = p1_head[mask_head]
        p2_head = p2_head[mask_head]
        print('p1_head', p1_head)
        print('p2_head', p2_head)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        # 绘制p1_head和p2_head的点并连线
        for i in range(len(p1_head)):
            # 画p1_head的点
            cv2.circle(pred, tuple(p1_head[i]), 3, (0, 255, 0), -1)  # 绿色点
            # 画p2_head的点
            cv2.circle(pred, tuple(p2_head[i]), 3, (0, 0, 255), -1)  # 蓝色点
            # 连接每一对点
            cv2.line(pred, tuple(p1_head[i]), tuple(p2_head[i]), (255, 0, 0), 2)  # 红色线

        # 显示图像
        cv2.namedWindow('Lines', cv2.WINDOW_NORMAL)
        cv2.imshow('Lines', pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.waitKey(0)
        # 429
        
        # if index % 2 == 0:
        #     head_x_coords, num = get_kde_x(pred, pts, 'head', 200) # 返回边缘x坐标列表， 用于与尾部计算偏移
        #     num_head = num
        #     # print(num_head, num_head)
        # else:
        #     # print('num_head', num_head)
        #     tail_x_coords, num = get_kde_x(pred, pts, 'tail', 200, num_head) # 返回边缘x坐标列表， 用于与头部计算偏移


        # p1_head = [[ 500 1753]
        # [ 510 1753]
        # [ 520 1753]
        # [ 530 1753]
        # [ 540 1753]
        # [ 550 1753]
        # [ 560 1753]
        # [ 570 1753]
        # [ 580 1753]
        # [ 590 1753]
        # [ 600 1747]]

        # p2_head = [[500 275]
        # [510 275]
        # [520 275]
        # [530 275]
        # [540 275]
        # [550 275]
        # [560 275]
        # [570 275]
        # [580 275]
        # [590 270]
        # [600 275]]

        # p3_head = [[ 560  509]
        # [ 550  588]
        # [ 540  653]
        # [ 530  699]
        # [ 520  777]
        # [ 510 1884]
        # [ 500 1890]
        # [ 490 1890]
        # [ 480 1890]
        # [ 470 1890]]

        # p4_head = [[560 414]
        # [550 407]
        # [540 400]
        # [530 400]
        # [520 400]
        # [510 400]
        # [500 400]
        # [490 400]
        # [480 400]
        # [470 400]]
