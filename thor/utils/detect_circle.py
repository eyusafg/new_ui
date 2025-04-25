import cv2
import numpy as np
import os


def detect_circle_(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = img_gray[0:2100, 0:3080]    

    edges = cv2.Canny(img_gray, 100, 200, apertureSize=3)
    # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 300,  param1=100, param2=15,  minRadius=50, maxRadius=60)

    if circles is not None:
        if len(circles[0]) == 1:
            return None, None
        else:
            circles = np.uint16(np.around(circles))
            max_circle = max(circles[0, :], key=lambda i: i[0])
            # min_circle = min(circles[0, :], key=lambda i: i[1])
            center = (max_circle[0], max_circle[1]) 
            radius = max_circle[2]

            return center, radius
    else:
        return None, None


def detect_circle(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lp = cv2.Laplacian(img_gray, -1, ksize=5)

    circles = cv2.HoughCircles(lp, cv2.HOUGH_GRADIENT, 1, 600,  param1=200, param2=25,  minRadius=50, maxRadius=60)

    x_f = None
    y_f = None
    r_f = None
    if circles is not None:
        # 将圆的数据转换为整数
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x = i[0]
            y = i[1]
            r = i[2]
            circle_region = img[y-r-20:y+r+20, x-r-20:x+r+20]
            circle_region_gray = cv2.cvtColor(circle_region, cv2.COLOR_BGR2GRAY)
            circles_ = cv2.HoughCircles(circle_region_gray, cv2.HOUGH_GRADIENT, 1, 100,  param1=100, param2=25,  minRadius=50, maxRadius=60)
            if circles_ is not None:
                circles_ = np.uint16(np.around(circles_))
                x_ = circles_[0, 0, 0]
                y_ = circles_[0, 0, 1]
                r_ = circles_[0, 0, 2]
                x_ori = x-r-20 + x_
                y_ori = y-r-20 + y_
                r_ori = r_
                print('r_ori:', r_ori)
                x_f = int((x_ori + x) / 2)
                y_f = int((y_ori + y) / 2)
                r_f = int((r_ori + r) / 2)
                break

                # for j in circles_[0, :]:
                #     cv2.circle(circle_region, (j[0], j[1]), j[2], (255, 255, 0), 2)

    if x_f is not None:
        # # 绘制外圆
        # cv2.circle(img, (int(x_f), int(y_f)), int(r_f), (0, 255, 0), 2)
        # # 绘制圆心
        # cv2.circle(img, (int(x_f), int(y_f)), 2, (0, 0, 255), 3)
        # # # 显示结果
        # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        # cv2.imshow("Result", img)
        # cv2.waitKey(0)
        return (x_f, y_f), r_f
    else:
        return None, None


if __name__ == '__main__':
    import os
    for im in os.listdir(r'Data\thor_kepoint\infer_result\DC_result'):
        im_p = os.path.join(r'Data\thor_kepoint\infer_result\DC_result', im)
        img = cv2.imread(im_p)
        c, r = detect_circle(img)
        print(c, r)
        # cv2.circle(img, c, 1, (0, 100, 100), 3)
        cv2.circle(img, c, r, (255, 0, 0), 3)
        cv2.namedWindow('circles', cv2.WINDOW_NORMAL)
        cv2.imshow('circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
