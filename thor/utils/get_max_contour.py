import cv2
import numpy as np

def get_max_contour(mask):
    contours, hirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(contours, key=cv2.contourArea)
    
    # 将轮廓点转为数组
    # approx_points = np.array([point[0] for point in max_cnt])
    contour_array = np.squeeze(max_cnt)

    return contour_array
