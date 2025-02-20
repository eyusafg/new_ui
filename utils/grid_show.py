import cv2
import numpy as np

def draw_grid(image, grid_size=16.5, bold_interval=5, grid_color=(0, 255, 0), axis_color=(0, 0, 255), thickness=1, bold_thickness=3, base_x=0, base_y=0):
    """
    在给定的图像上以特定的基点绘制网格，基点的x轴和y轴以红色显示。
    
    :param image: 输入的图像。
    :param grid_size: 网格的宽度和高度。
    :param bold_interval: 加粗线条的间隔。
    :param grid_color: 网格的颜色。
    :param axis_color: 基点x轴和y轴的颜色。
    :param thickness: 普通网格线的厚度。
    :param bold_thickness: 加粗网格线的厚度。
    :param base_x: 网格基点的X坐标。
    :param base_y: 网格基点的Y坐标。
    :return: 绘制了网格的图像。
    """
    height, width, _ = image.shape
    
    # 计算需要绘制的水平和垂直线的数量
    num_horizontal_lines = int(np.ceil(height / grid_size))
    num_vertical_lines = int(np.ceil(width / grid_size))
    
    # 计算基点左侧和上方最近的网格线位置
    start_x = int(base_x - (base_x // grid_size) * grid_size)
    start_y = int(base_y - (base_y // grid_size) * grid_size)
    
    # 绘制水平线
    for i in range(num_horizontal_lines + 1):
        y = int(start_y + i * grid_size)
        if y < 0 or y > height: continue
        if i % bold_interval == (base_y - start_y)%(bold_interval*grid_size) // grid_size:
            cv2.line(image, (0, y), (width, y), grid_color, bold_thickness)
        else:
            cv2.line(image, (0, y), (width, y), grid_color, thickness)
    
    # 绘制垂直线
    for j in range(num_vertical_lines + 1):
        x = int(start_x + j * grid_size)
        if x < 0 or x > width: continue
        if j % bold_interval == (base_x - start_x)%(bold_interval*grid_size) // grid_size:
            cv2.line(image, (x, 0), (x, height), grid_color, bold_thickness)
        else:
            cv2.line(image, (x, 0), (x, height), grid_color, thickness)
    
    # 绘制基点的x轴和y轴
    cv2.line(image, (base_x, 0), (base_x, height), axis_color, bold_thickness-1)  # Y轴
    cv2.line(image, (0, base_y), (width, base_y), axis_color, bold_thickness-1)  # X轴

    return image


if __name__ == '__main__':
    # 读取原始图像

    ori_frame = cv2.imread(r'Data\thor_keypoint_data\2024_11_27_16_20_48.png')

    # 定义基点
    base_x = 2084
    base_y = 972

    # 绘制网格
    grid_image = draw_grid(ori_frame, base_x=base_x, base_y=base_y)
    cv2.imwrite('grid_image1.jpg', grid_image)
    # 显示图像
    cv2.namedWindow('Grid Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Grid Image', grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 如果需要保存图像
    # cv2.imwrite('grid_image.jpg', grid_image)
