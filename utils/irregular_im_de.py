import cv2
import numpy as np
from scipy.stats import gaussian_kde

def find_nearest_point(point, points):
    distances = np.linalg.norm(points - point, axis=1)
    nearest_index = np.argmin(distances)
    return points[nearest_index]

def get_kde(mask, points):

    p1 = points[0]
    p2 = points[1]

    # find contours
    contours, hirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(contours, key=cv2.contourArea)
    # arlength = cv2.arcLength(max_cnt, True)
    # approx = cv2.approxPolyDP(max_cnt, 0.0002 * arlength, True)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 将轮廓点转为数组
    approx_points = np.array([point[0] for point in max_cnt])

    if np.any(np.all(approx_points == p1, axis=1)):
        idx1 = np.where(np.all(approx_points == p1, axis=1))[0][0]
    else:
        p1 = find_nearest_point(p1, approx_points)
        idx1 = np.where(np.all(approx_points == p1, axis=1))[0][0]

    if np.any(np.all(approx_points == p2, axis=1)):
        idx2 = np.where(np.all(approx_points == p2, axis=1))[0][0]
    else:
        p2 = find_nearest_point(p2, approx_points)
        idx2 = np.where(np.all(approx_points == p2, axis=1))[0][0]

    # 确保 idx1 小于 idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    # 获取 p1 和 p2 之间的轮廓点集合
    pb1 = approx_points[idx1:idx2 + 1]
    # 获取剩余的轮廓点集合
    remaining_points_1 = approx_points[:idx1]
    remaining_points_2 = approx_points[idx2+1:]
    pb2 = np.concatenate((remaining_points_1, remaining_points_2), axis=0) if remaining_points_1.size != 0 or remaining_points_2.size != 0 else np.array([])

    # 比较两个集合的数量并选择较少的那个
    if len(pb1) <= len(pb2):
        points_between = pb1
    else:
        points_between = pb2

    # 绘制轮廓点
    # for point in approx_points:
    #     cv2.circle(mask_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    # 绘制 p1 和 p2
    # cv2.circle(mask_bgr, (int(p1[0]), int(p1[1])), 5, (0, 255, 0), -1)
    # cv2.circle(mask_bgr, (int(p2[0]), int(p2[1])), 5, (0, 255, 0), -1)

    # 绘制 p1 和 p2 之间的轮廓点
    # for point in points_between:
    #     cv2.circle(mask_bgr, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)


    # 取线段的x值
    x_coords = points_between[:, 0]
    print("X coordinates:", x_coords)   

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


    # 滤波
    # window_size = 5  # 滤波窗口大小
    # filtered_x_coords_medfilt = medfilt(x_coords, kernel_size=window_size)
    # print("Median filtered X coordinates:", filtered_x_coords_medfilt)

    # 聚类
    # db = DBSCAN(eps=0.3, min_samples=10).fit(points_between)
    # labels = db.labels_

    # # 获取最大簇的索引
    # max_cluster_label = max(set(labels), key=list(labels).count)
    # inliers = labels == max_cluster_label
    # print("Inlier X coordinates from DBSCAN:", points_between[inliers, 0])
    # mean_x_inliers = np.mean(points_between[inliers, 0])
    # cv2.line(mask_bgr, (int(mean_x_inliers), int(point[1])),(int(mean_x_inliers), int(int(p1[1]))), (255, 0, 255), 2)

    # 最小二乘
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


    # 基于密度/直方图  更贴近实际情况       ！！！！！！！！！！！！！！！！！！！！！
    density = gaussian_kde(x_coords)
    xs = np.linspace(min(x_coords), max(x_coords), 20)
    density_estimation = density(xs)
    max_density_index = np.argmax(density_estimation)
    max_density_x = xs[max_density_index]
    print(f"Maximum density is at x={max_density_x}")
    # cv2.circle(mask_bgr, (int(max_density_x), int(point[1])), 10, (255, 0, 255), -1)
    # cv2.line(mask_bgr, (int(max_density_x), int(point[1])),(int(max_density_x), int(int(p1[1]))), (255, 0, 255), 1)

    # plt.plot(xs, density_estimation)
    # plt.fill_between(xs, density_estimation, alpha=0.5)
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Density')
    # plt.title('Kernel Density Estimation of X Coordinates')
    # plt.show()

    # for i in approx:
    #     cv2.circle(mask_bgr, (int(i[0][0]), int(i[0][1])), 5, (0, 0, 255), -1)
    # cv2.circle(mask_bgr, (int(p1[0]), int(p1[1])), 5, (0, 255, 0), -1)
    # cv2.circle(mask_bgr, (int(p2[0]), int(p2[1])), 5, (0, 255, 0), -1)

    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("mask", mask_bgr)
    # cv2.waitKey(0)
    return max_density_x
