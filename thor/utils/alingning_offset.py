import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib


class AlignmentOptimizer:
    def __init__(self, head_x_coords, tail_x_coords, roi_rotate_base_p_x, roi_tail_rotate_p_x, is_V=False):

        """
        初始化对齐优化器
        :param front_offsets: 前部偏移列表 (长度50)
        :param rear_offsets: 尾部偏移列表 (长度50)
        """
     
        self.roi_rotate_base_p_x = roi_rotate_base_p_x
        self.roi_tail_rotate_p_x = roi_tail_rotate_p_x
        self.head_x_coords = self.preprocess_data(head_x_coords)
        self.tail_x_coords = self.preprocess_data(tail_x_coords)
        front_offsets = [x - roi_rotate_base_p_x for x in head_x_coords]
        rear_offsets = [x - roi_tail_rotate_p_x for x in tail_x_coords]
        self.front = self.preprocess_data(front_offsets)
        self.rear = self.preprocess_data(rear_offsets)

        self.aligned = self.front + self.rear  # 对齐后偏移值
        self.aligned = self.aligned.astype(np.float32)
        self.original_indices = np.argsort(self.aligned)
        self.aligned = np.sort(self.aligned)
        self.make_unique()
        self.cumsum = np.cumsum(self.aligned)  # 累积和数组
        # print('self.cumsum', self.cumsum)
        self.total = self.cumsum[-1]  # 总偏移量
        self.optimal_idx = None
        self.is_V = is_V
        self.total_flag = False

    def get_max_offset(self):
        """获取最大偏移值"""
        max_offset = self.aligned[-1]
        max_front = np.max(self.front)
        rear = max_offset - max_front
        return max_front, rear
    
    def make_unique(self):
        """确保 self.aligned 中的所有值都是唯一的"""
        epsilon = 0.01
        for i in range(1, len(self.aligned)):
            if self.aligned[i] <= self.aligned[i-1]:
                self.aligned[i] = self.aligned[i-1] + epsilon
               
    @staticmethod
    def preprocess_data(data):
        array_data = np.array(data)
        # 计算步长
        step = len(array_data) // 50
        # 生成等分的索引
        indices = np.arange(0, len(array_data), step)
        # 确保我们取到正好 50 个点
        if len(indices) > 50:
            indices = indices[:50]
        elif len(indices) < 50:
            indices = np.append(indices, [indices[-1]] * (50 - len(indices)))
        x_points = array_data[indices]
        return x_points
    
    def find_optimal_split(self):
        """寻找使左右差值最小的分割点"""
        min_diff = float('inf')
        best_k = 0
        current_diff_list = []
        for k in range(len(self.aligned)-1):
            left_sum = self.cumsum[k] - (k+1) * self.aligned[k+1]
            left_sum_abs = abs(left_sum)
            right_sum = (self.total - self.cumsum[k]) - (50-(k+1)) * self.aligned[k+1]
            right_sum_abs = abs(right_sum)
            # current_diff = abs(left_sum - right_sum)
            current_diff = abs(left_sum_abs - right_sum_abs)
            current_diff_ = left_sum_abs - right_sum_abs
            current_diff_list.append(current_diff_)
            if current_diff < min_diff:
                min_diff = current_diff
                best_k = k
        print('current_diff_list', current_diff_list)
        self.optimal_idx = best_k + 1
        original_best_k = self.original_indices[self.optimal_idx]
        self.optimal_left_offset = self.cumsum[self.optimal_idx-1] - (self.optimal_idx) * self.cumsum[self.optimal_idx]
        self.optimal_right_offset = (self.total - self.optimal_left_offset) - (50-(self.optimal_idx)) * self.cumsum[self.optimal_idx]
        self.optimal_left_offset = abs(self.optimal_left_offset)
        self.optimal_right_offset = abs(self.optimal_right_offset)
        circle_head_x = self.head_x_coords[original_best_k]
        circle_tail_x = self.tail_x_coords[original_best_k] 
        front_offset = self.front[original_best_k]
        rear_offset = self.rear[original_best_k]
        t = front_offset + rear_offset
        # print('circle_head_x', circle_head_x)
        # print('circle_tail_x', circle_tail_x)
        '''
        需要从对齐偏移中找到最小35个， 然后求最大 最小只差小于一定阈值
        然后这35个最小的尾部x值以及对应得头部得值
        需要防止可能存在多个存在多个相同得最小得x 
        '''

        # 将current_diff与索引配对并按绝对值排序
        indexed_diffs = sorted(
            [(abs(val), k, val) for k, val in enumerate(current_diff_list)],
            key=lambda x: x[0]  # 按绝对值排序
        )[:30]  # 取绝对值最小的35个
        # 提取这35个元素的原始current_diff值（包含正负）
        original_diffs = [val for (abs_val, k, val) in indexed_diffs]
        print('original_diffs',original_diffs)
        # 计算这35个原始值的极差
        max_sn_15 = max(original_diffs)
        min_sn_15 = min(original_diffs)

        '''
        0506 修改2500为3500
        0513传入的前部和尾部的坐标事先经过y排序， 确保前部和尾部一一对应， 阈值改为2000
        '''
        front_offset_min = None
        rear_offset_min = None
        circle_head_x_min = None
        circle_tail_x_min = None
        if max_sn_15 - min_sn_15 <= 1200 and self.is_V is False:
            self.total_flag = True
            print(66666666666666666666666666666666666666666666666666666)

            # 提取原始k值（注意这里k是原始分割点索引）
            candidate_k_values = [k for (_, k, _) in indexed_diffs]
            
            # 转换为排序后的aligned索引（k对应optimal_idx=k+1）
            candidate_optimal_indices = [k+1 for k in candidate_k_values]
            
            # 获取原始坐标索引（关键映射步骤）
            original_candidates = self.original_indices[candidate_optimal_indices]
            
            # 获取尾部坐标并找最小值
            tail_x_candidates = self.rear[original_candidates]

            min_tail_x = np.min(tail_x_candidates)

            min_tail_mask = tail_x_candidates == min_tail_x

            min_tail_original = original_candidates[min_tail_mask]
            
            # 处理多个候选的情况
            if len(min_tail_original) > 0:
                print(1111111111111111111)
                # 获取这些候选点在aligned数组中的值
                rear_values = self.rear[min_tail_original]
                # 找到最小aligned值的索引
                min_rear_idx = np.argmin(rear_values)
                # 最终确定最佳原始索引
                original_best_k = min_tail_original[min_rear_idx]
                
        elif self.is_V:
            '''
            V型取最小值
            '''
            indices_min = np.argmin(self.rear)
            rear_offset_min = self.rear[indices_min]
            front_offset_min = t - rear_offset_min
            circle_head_x_min = front_offset_min + self.roi_rotate_base_p_x
            circle_tail_x_min = self.tail_x_coords[indices_min]

        # front_offset = self.front[original_best_k]
        rear_offset = self.rear[original_best_k]
        front_offset = t - rear_offset
        # circle_head_x_ = self.head_x_coords[original_best_k]
        circle_head_x_ = front_offset + self.roi_rotate_base_p_x
        circle_tail_x_ = self.tail_x_coords[original_best_k]         
        if front_offset_min is None:
            return front_offset, rear_offset, circle_head_x, circle_tail_x, circle_head_x_, circle_tail_x_, self.total_flag
        else:
            return front_offset_min, rear_offset_min, circle_head_x,circle_tail_x, circle_head_x_min, circle_tail_x_min, self.total_flag
    
    # def visualize(self):
    #     """可视化对齐偏移值与最优分割点"""
    #     # 设置默认字体为支持中文的字体
    #     matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    #     matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    #     plt.figure(figsize=(10,6))
        
    #     # 绘制对齐后的偏移值
    #     plt.subplot(211)
    #     plt.bar(range(50), self.aligned, alpha=0.7, label='对齐后偏移值')
    #     plt.axvline(self.optimal_idx, color='red', linestyle='--', 
    #                 label=f'最优分割点 (k={self.optimal_idx})')
    #     plt.ylabel('偏移值')
    #     plt.legend()
        
    #     # 标注左右区域
    #     plt.annotate(f'左区域和: {self.optimal_left_offset:.1f}',
    #                 xy=(self.optimal_idx//2, max(self.aligned)*0.8),
    #                 fontsize=10, color='darkblue')
    #     plt.annotate(f'右区域和: {self.optimal_right_offset:.1f}',
    #                 xy=(self.optimal_idx + 10, max(self.aligned)*0.6),
    #                 fontsize=10, color='darkgreen')
        
    #     # 绘制累积和曲线
    #     plt.subplot(212)
    #     plt.plot(self.cumsum, 'b-', lw=2, label='累积和')
    #     plt.axvline(self.optimal_idx, color='red', linestyle='--')
    #     plt.xlabel('索引')
    #     plt.ylabel('累积偏移和')
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
        
    #     plt.suptitle(f"最优分割点: k={self.optimal_idx} (最小差值={abs(2*self.cumsum[self.optimal_idx]-self.total):.1f})")
    #     plt.tight_layout()
    #     plt.show()

# 示例用法 -------------------------------------------------
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    front = np.random.randint(-10,5,50)
    rear = np.random.randint(-10,5,50)
    
    # 创建优化器
    optimizer = AlignmentOptimizer(front, rear, 5, 5)
    front_offset, rear_offset, min_diff, k = optimizer.find_optimal_split()
    
    # 可视化
    # optimizer.visualize()
    print(f"最优分割点: {k}, 最小差值: {min_diff:.1f}")
