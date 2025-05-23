import cv2  
import numpy as np  



def get_contour_corner(max_cn, orgin_img, flag, vis):
    # print('max_cn', max_cn)
    # print('orgin_img', orgin_img.shape)
    x, y, w, h = cv2.boundingRect(max_cn)
    cv2.rectangle(orgin_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    cnt = cv2.approxPolyDP(max_cn, 0.0025*cv2.arcLength(max_cn, True), True)
    cnt_np = np.squeeze(cnt)

    if flag == 'start':
        '''
        先取轮廓中心的x， 取小于这个x的点集
        取y值最小以及y值最大且x值最小的两个点作为关键点
        有一种情况， 在角点边缘可能聚集了两个点， 很近， 需要在进行判断， x和y是不是都很近， x比所选的小， y相距10-20个像素
        '''
        # M = cv2.moments(cnt)
        # # 使用矩计算中点
        # center_x = int(M['m10'] / M['m00'])
        '''
        对center_x 还需要做一个限制， 最大边缘是多少
        '''
        # center_x = x + 1/3*w  # 这个阈值应该通过外部输入
        center_x = x + 168  # 这个阈值应该通过外部输入
        cnt_np = cnt_np[cnt_np[:, 0] < center_x]
  
        min_y_index = np.argmin(cnt_np[:, 1])
        min_y_point = cnt_np[min_y_index]

        # 找到 y 值最大的点，并在这些点中找到 x 值最小的点
        max_y_value = np.max(cnt_np[:, 1])
        max_y_indices = np.where(cnt_np[:, 1] == max_y_value)[0]
        max_y_points = cnt_np[max_y_indices]
        min_x_index_among_max_y_points = np.argmin(max_y_points[:, 0])
        max_y_min_x_point = max_y_points[min_x_index_among_max_y_points]

        # 筛选与 min_y_point 的 y 值相差在 40 以内的点，并且 x 值小于 min_y_point 的 x 值
        min_y_threshold = min_y_point[1] - 30
        max_y_threshold = min_y_point[1] + 30
        filtered_by_min_y = cnt_np[(cnt_np[:, 1] >= min_y_threshold) & (cnt_np[:, 1] <= max_y_threshold) & (cnt_np[:, 0] < min_y_point[0])]
        print('filtered_by_min_y', filtered_by_min_y)
        # 筛选与 max_y_min_x_point 的 y 值相差在 40 以内的点，并且 x 值小于 max_y_min_x_point 的 x 值
        min_y_threshold = max_y_min_x_point[1] - 30
        max_y_threshold = max_y_min_x_point[1] + 30
        filtered_by_max_y_min_x = cnt_np[(cnt_np[:, 1] >= min_y_threshold) & (cnt_np[:, 1] <= max_y_threshold) & (cnt_np[:, 0] < max_y_min_x_point[0])]
    else:
        center_x = x + w - 168
        cnt_np = cnt_np[cnt_np[:, 0] > center_x]
        # 找到 y 值最小的点
        min_y_index = np.argmin(cnt_np[:, 1])
        min_y_point = cnt_np[min_y_index]

        max_y_value = np.max(cnt_np[:, 1])
        max_y_indices = np.where(cnt_np[:, 1] == max_y_value)[0]
        max_y_points = cnt_np[max_y_indices]
        max_x_in_max_y_points = np.argmax(max_y_points[:, 0])
        max_y_max_x_point = max_y_points[max_x_in_max_y_points]

        # 筛选与 min_y_point 的 y 值相差在 40 以内的点，并且 x 值小于 min_y_point 的 x 值
        min_y_threshold = min_y_point[1] - 30 
        max_y_threshold = min_y_point[1] + 30
        filtered_by_min_y = cnt_np[(cnt_np[:, 1] >= min_y_threshold) & (cnt_np[:, 1] <= max_y_threshold) & (cnt_np[:, 0] > min_y_point[0])]
        print('filtered_by_min_y', filtered_by_min_y)
        # 筛选与 max_y_min_x_point 的 y 值相差在 40 以内的点，并且 x 值小于 max_y_min_x_point 的 x 值
        min_y_threshold = max_y_max_x_point[1] - 30
        max_y_threshold = max_y_max_x_point[1] + 30
        filtered_by_max_y_min_x = cnt_np[(cnt_np[:, 1] >= min_y_threshold) & (cnt_np[:, 1] <= max_y_threshold) & (cnt_np[:, 0] > max_y_max_x_point[0])]
    if vis:
        for i in cnt:
            cv2.circle(orgin_img, (int(i[0][0]), int(i[0][1])), 10, (0, 0, 255), -1)
        cv2.line(orgin_img, (int(center_x), 0), (int(center_x), 800), (255, 0, 0), 5)
        if filtered_by_min_y.size > 0:
            cv2.circle(orgin_img, (int(filtered_by_min_y[0][0]), int(filtered_by_min_y[0][1])), 5, (255, 0, 0), -1)  # y 值最小的点
        else:
            cv2.circle(orgin_img, (int(min_y_point[0]), int(min_y_point[1])), 5, (255, 0, 0), -1)  # y 值最小的点
        if filtered_by_max_y_min_x.size > 0:
            cv2.circle(orgin_img, (int(filtered_by_max_y_min_x[0][0]), int(filtered_by_max_y_min_x[0][1])), 5, (0, 255, 0), -1)  # y 值最大且 x 值最小的点
        else:
            cv2.circle(orgin_img, (int(max_y_min_x_point[0]), int(max_y_min_x_point[1])), 5, (0, 255, 0), -1)  # y 值最大且 x 值最小的点
        cv2.imshow('orgin_img', orgin_img)
        cv2.waitKey(0)
    if filtered_by_min_y.size > 0 and filtered_by_max_y_min_x.size > 0:
        # return [[filtered_by_min_y[0][0], filtered_by_min_y[0][1]], filtered_by_max_y_min_x[0][1]]
        return [[filtered_by_min_y[0][0], filtered_by_min_y[0][1]], [filtered_by_max_y_min_x[0][0], filtered_by_max_y_min_x[0][1]]]
    elif filtered_by_min_y.size > 0 and filtered_by_max_y_min_x.size == 0 and flag == 'start':
        return [[filtered_by_min_y[0][0],filtered_by_min_y[0][1]], [max_y_min_x_point[0], max_y_min_x_point[1]]]
    elif filtered_by_min_y.size > 0 and filtered_by_max_y_min_x.size == 0 and flag == 'end':
        return [[max_y_max_x_point[0], max_y_max_x_point[1]], [filtered_by_min_y[0][0], filtered_by_min_y[0][1]]]
    elif filtered_by_max_y_min_x.size > 0 and filtered_by_min_y.size == 0 and flag == 'start':
        return [[min_y_point[0][0], min_y_point[0][1]], [filtered_by_max_y_min_x[0][0], filtered_by_max_y_min_x[0][1]]]
    elif filtered_by_max_y_min_x.size > 0 and filtered_by_min_y.size == 0 and flag == 'end':
        return [min_y_point[0][0], min_y_point[0][1]], [filtered_by_max_y_min_x[0][0], filtered_by_max_y_min_x[0][1]]
    else:
        return None

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

if __name__ == '__main__':
    import onnxruntime as rt
    import os
    from get_max_contour import get_max_contour

    segm_model_path = r'models\thor_segm_20250424_model_69.onnx'
    segm_sess, segm_input_name, segm_output_name = load_model(segm_model_path, delay=False)
    im_p = r'C:\Data\thor_kepoint\error_detect\20250515\2025_05_15_22_16_35_start.png'
    im = cv2.imread(im_p)
    im_ = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)) 
    se = infer_segm(im, segm_sess, segm_input_name, segm_output_name)
    pred = np.where(se.squeeze() > 0, 255, 0).astype(np.uint8)
    pred = cv2.resize(pred, (im.shape[1],im.shape[0]))      
    max_contour = get_max_contour(pred)
    pts = get_contour_corner(max_contour, im, 'start', True)
    pts = np.array(pts).astype(np.float64)
    pts[:, 1] *= 1.5
    b = abs(abs(pts[0][1]) - abs(pts[1][1])) * 0.042
    print(b)
    if pts.ndim == 2:
        print('维度正常')
    if np.any(pts[:, 0]<207):
        print('所有x坐标小于夹爪边界')
    elif np.any(pts[:, 0]>593): 
        print('所有x坐标大于夹爪边界')
    elif abs(abs(pts[0][0]) - abs(pts[1][0])) > 350:
        print('x轴偏差过大')
    elif np.all(pts < 0):
        print('所有坐标小于0')
    for i in range(len(pts)):
        cv2.circle(im, (int(pts[i][0]), int(pts[i][1])), 5, (0, 0, 255), -1)
    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    print(pts)

