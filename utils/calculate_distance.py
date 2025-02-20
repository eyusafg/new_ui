import cv2
import numpy as np
import os 
import pandas as pd


def  findcorners(img, new_im,show):
    name = 'points'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chessboard_size = (41, 41) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        cv2.cornerSubPix(gray, corners, (9,9), (-1,-1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
  
        corner_coords = corners.reshape(-1, 2)

        for i, corner in enumerate(corner_coords):
            print(f"corner {i}: {corner}")

            cv2.putText(img, str(i), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        if show:
            cv2.namedWindow('Chessboard corners', cv2.WINDOW_NORMAL)
            cv2.imshow('Chessboard corners', img)
            cv2.waitKey(10)
        np.savetxt(f'{os.path.join(new_im, name)}.txt', corner_coords, fmt='%f', header='X Y', comments='')
    else:
        print(img)   
        # shutil.move(img, new_im) 
        print("Chessboard not found")

def calculate_distance(img):
    
    points_save_path = 'profiles//chessboard_points'
    if not os.path.exists(points_save_path):
        os.makedirs(points_save_path)

    # 找角点
    findcorners(img, points_save_path, True)   

    all_points = []
    with open(os.path.join(points_save_path, 'points.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = map(float, line.strip().split())
            all_points.append((x, y))

    groups = [all_points[i:i+41] for i in range(0, len(all_points), 41)]

    horizontal_results = []
    vertical_results = []

    # 横向
    for i in range(0, len(groups)-5, 5):
        group_a = groups[i]
        group_b = groups[i+5]
        for j in range(len(group_a)):
            point_a = group_a[j]
            point_b = group_b[j]
            w_distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
            print(f"The distance between the two points is {w_distance} pixels.")
            # w_real_distance = 0.041 * w_distance
            w_conversion_factor = 10 / w_distance
            horizontal_results.append({'Group': f'Between Group {i} and Group {i + 5}', 'Distance (pixels)': w_distance, 'conversion factor (units)': w_conversion_factor})
            # cv2.circle(im, (int(point_a[0]), int(point_a[1])), 10, (0, 0, 255), -1)
            # cv2.circle(im, (int(point_b[0]), int(point_b[1])), 10, (0, 255, 0), -1)

    # 竖向
    for i in range(0, len(lines), 41):
        for j in range(i, i+41, 5):
            if (j+1) % 41 == 0:
                continue
            p1 = lines[j].strip().split()
            p1_x, p1_y = float(p1[0]), float(p1[1])
            if j + 5 > len(lines):
                break
            p2 = lines[j+5].strip().split()
            p2_x, p2_y = float(p2[0]), float(p2[1])
            h_distance = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
            print(f"The distance between the two points is {h_distance} pixels.")
            # h_real_distance = 0.041 * h_distance
            h_conversion_factor = 10 / h_distance
            vertical_results.append({'Group': f'Between Point {j} and Point {j + 5}', 'Distance (pixels)': h_distance, 'conversion factor  (10mm)': h_conversion_factor})

            # cv2.circle(im, (int(p1_x), int(p1_y)), 10, (0, 0, 255), -1)
            # cv2.circle(im, (int(p2_x), int(p2_y)), 10, (0, 255, 0), -1)

            # cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
            # cv2.imshow('Corners', im)
            # cv2.waitKey(0)
        
    # 创建 DataFrame
    horizontal_df = pd.DataFrame(horizontal_results)
    vertical_df = pd.DataFrame(vertical_results)

    max_length = max(len(horizontal_df), len(vertical_df))
    horizontal_df = horizontal_df.reindex(range(max_length))
    vertical_df = vertical_df.reindex(range(max_length))

    # 并排合并
    result_df = pd.concat([horizontal_df.reset_index(drop=True), vertical_df.reset_index(drop=True)], axis=1)

    # 将列名调整为更清晰的描述
    result_df.columns = ['横向', 
                        'Horizontal Distance (pixels)', 
                        'Horizontal Real Distance (10mm)', 
                        '竖向', 
                        'Vertical Distance (pixels)', 
                        'Vertical Real Distance (10mm)']
    # 写入 CSV 文件
    result_df.to_csv('distance_results.csv', index=False)
    print("结果已写入到 distance_results.csv")

    conversion_factors = [result['conversion factor (units)'] for result in horizontal_results]
    average_conversion_factor = round(sum(conversion_factors) / len(conversion_factors), 3)
    return average_conversion_factor


if __name__ == '__main__':
    

    points_save_path = 'points_0107'
    if not os.path.exists(points_save_path):
        os.makedirs(points_save_path)
    im_path = 'images'

    points_txt_list = os.listdir(points_save_path)
    points_txt_list_ = [txt for txt in points_txt_list if os.path.isfile(os.path.join(points_save_path, txt))]

    # # 找角点
    for im in os.listdir(im_path):
        if im.split('.')[0] + '.txt' in points_txt_list_:
            break
        img = os.path.join(im_path, im)
        findcorners(img, points_save_path, True)   

    im = cv2.imread('images/2025_01_07_10_23_36.png')
    for txt in points_txt_list_:
        all_points = []
        with open(os.path.join(points_save_path, txt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                x, y = map(float, line.strip().split())
                all_points.append((x, y))

        groups = [all_points[i:i+41] for i in range(0, len(all_points), 41)]

        horizontal_results = []
        vertical_results = []

        # 横向
        for i in range(0, len(groups)-5, 5):
            group_a = groups[i]
            group_b = groups[i+5]
            for j in range(len(group_a)):
                point_a = group_a[j]
                point_b = group_b[j]
                w_distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
                print(f"The distance between the two points is {w_distance} pixels.")
                w_real_distance = 0.041 * w_distance
                horizontal_results.append({'Group': f'Between Group {i} and Group {i + 5}', 'Distance (pixels)': w_distance, 'Real Distance (units)': w_real_distance})
                # cv2.circle(im, (int(point_a[0]), int(point_a[1])), 10, (0, 0, 255), -1)
                # cv2.circle(im, (int(point_b[0]), int(point_b[1])), 10, (0, 255, 0), -1)

        # 竖向
        for i in range(0, len(lines), 41):
            for j in range(i, i+41, 5):
                if (j+1) % 41 == 0:
                    continue
                p1 = lines[j].strip().split()
                p1_x, p1_y = float(p1[0]), float(p1[1])
                if j + 5 > len(lines):
                    break
                p2 = lines[j+5].strip().split()
                p2_x, p2_y = float(p2[0]), float(p2[1])
                h_distance = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
                print(f"The distance between the two points is {h_distance} pixels.")
                h_real_distance = 0.041 * h_distance
                vertical_results.append({'Group': f'Between Point {j} and Point {j + 5}', 'Distance (pixels)': h_distance, 'Real Distance (10mm)': h_real_distance})

                # cv2.circle(im, (int(p1_x), int(p1_y)), 10, (0, 0, 255), -1)
                # cv2.circle(im, (int(p2_x), int(p2_y)), 10, (0, 255, 0), -1)

                # cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
                # cv2.imshow('Corners', im)
                # cv2.waitKey(0)
            
        # 创建 DataFrame
        horizontal_df = pd.DataFrame(horizontal_results)
        vertical_df = pd.DataFrame(vertical_results)

        # # 合并 DataFrame
        # result_df = pd.DataFrame(columns=['Direction', 'Distance (pixels)', 'Real Distance (10mm)'])
        # result_df = pd.concat([result_df, horizontal_df.assign(Direction='横向')])
        # result_df = pd.concat([result_df, vertical_df.assign(Direction='纵向')])

        max_length = max(len(horizontal_df), len(vertical_df))
        horizontal_df = horizontal_df.reindex(range(max_length))
        vertical_df = vertical_df.reindex(range(max_length))

        # 并排合并
        result_df = pd.concat([horizontal_df.reset_index(drop=True), vertical_df.reset_index(drop=True)], axis=1)

        # 将列名调整为更清晰的描述
        result_df.columns = ['横向', 
                            'Horizontal Distance (pixels)', 
                            'Horizontal Real Distance (10mm)', 
                            '竖向', 
                            'Vertical Distance (pixels)', 
                            'Vertical Real Distance (10mm)']
        # 写入 CSV 文件
        result_df.to_csv('distance_results.csv', index=False)
        print("结果已写入到 distance_results.csv")





