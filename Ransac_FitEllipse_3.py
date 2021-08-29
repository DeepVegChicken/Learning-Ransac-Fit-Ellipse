import cv2
import math
import random
import numpy as np
from numpy.linalg import inv, svd, det


class RANSAC:
    def __init__(self, data, items, threshold, P, S):
        self.point_data = data  # 数据点
        self.items = items  # 迭代次数
        self.threshold = threshold  # 数据和模型之间可接受的差值
        self.P = P  # 希望的得到正确模型的概率
        self.size = len(self.point_data) * S  # 判断是否当前模型已经符合超过预设点
        self.label = False

        self.count = 0
        self.best_model = None

    def random_sampling(self, n):
        """随机取n个数据点"""
        all_point = self.point_data
        np.random.shuffle(all_point)
        select_point = all_point[:n]
        return select_point

    def make_model(self, select_point):
        # create ellipse model
        point1, point2, point3, point4, point5 = select_point

        MatA = np.matrix([
            [point1[0] * point1[0], point1[0] * point1[1], point1[1] * point1[1], point1[0], point1[1], 1],
            [point2[0] * point2[0], point2[0] * point2[1], point2[1] * point2[1], point2[0], point2[1], 1],
            [point3[0] * point3[0], point3[0] * point3[1], point3[1] * point3[1], point3[0], point3[1], 1],
            [point4[0] * point4[0], point4[0] * point4[1], point4[1] * point4[1], point4[0], point4[1], 1],
            [point5[0] * point5[0], point5[0] * point5[1], point5[1] * point5[1], point5[0], point5[1], 1]])

        U, D, V = svd(MatA)
        VT = np.transpose(V)

        V05 = VT[0, 5]
        if math.fabs(V05) < 0.000001:
            return 0

        A = 1
        B = VT[1, 5] / V05
        C = VT[2, 5] / V05
        D = VT[3, 5] / V05
        E = VT[4, 5] / V05
        F = VT[5, 5] / V05

        thres1 = 4 * A * C - B * B
        ellParamMat = np.matrix([[2 * A, B, D], [B, 2 * C, E], [D, E, 2 * F]])
        thres2 = det(ellParamMat) * (A + C)
        if (thres1 <= 0) or (thres2 >= 0):
            return 0

        # get ellipse param
        x = (B * E - 2 * C * D) / (4 * A * C - B * B)
        y = (B * D - 2 * A * E) / (4 * A * C - B * B)

        if abs(B) <= 0.0001 and A < C:
            Angle = 0
        elif abs(B) <= 0.0001 and A > C:
            Angle = 90
        elif A < C:
            Angle = 0.5 * math.atan(B / (A - C)) * 180 / math.pi
        else:
            Angle = 90 + 0.5 * math.atan(B / (A - C)) * 180 / math.pi

        epTemp1 = A * x * x + C * y * y + B * x * y - F
        epTemp2 = A + C
        epTemp3 = math.sqrt((A - C) * (A - C) + B * B)
        LAxis = math.sqrt(2 * epTemp1 / (epTemp2 - epTemp3))
        SAxis = math.sqrt(2 * epTemp1 / (epTemp2 + epTemp3))

        return x, y, LAxis, SAxis, Angle, [A, B, C, D, E, F]

    def eval_model(self, model):
        Count = 0
        x, y, LAxis, SAxis, Angle, EllipseFunc_Param = model  # model param
        A, B, C, D, E, F = EllipseFunc_Param  # elliptic equation coefficient

        # calculate the ellipse focus
        Axis = math.sqrt(LAxis * LAxis - SAxis * SAxis)
        f1_x = x - Axis * math.cos(Angle * math.pi / 180)
        f1_y = y - Axis * math.sin(Angle * math.pi / 180)
        f2_x = x + Axis * math.cos(Angle * math.pi / 180)
        f2_y = y + Axis * math.sin(Angle * math.pi / 180)

        F1F2 = math.sqrt((f1_x - f2_x) ** 2 + (f1_y - f2_y) ** 2)
        if (2 * LAxis) <= F1F2:  # should: 2a > |F1F2|
            print('The model does not satisfy the definition of elliptic equation')
            return 0

        for point in self.point_data:
            dist1 = math.sqrt((point[0] - f1_x) ** 2 + (point[1] - f1_y) ** 2)
            dist2 = math.sqrt((point[0] - f2_x) ** 2 + (point[1] - f2_y) ** 2)
            dist = dist1 + dist2

            # |PF1| + |PF2| = 2a
            if math.fabs(dist - 2 * LAxis) < self.threshold:
                Count += 1
        return Count

    def execute_ransac(self):
        while self.items > 0:
            # random select 5 points
            select_point = self.random_sampling(n=5)

            # create model and get ellipse param
            model = self.make_model(select_point)
            if model == 0:
                continue

            # eval model and calculate number of inter points
            Count = self.eval_model(model)

            # number of new inter points more than number of old inter points
            if Count > self.count:
                self.count = Count  # Update internal points
                self.best_model = model  # Save the best model
                # Update the number of iterations dynamically
                self.items = int(math.log(1 - self.P) / math.log(1 - pow(Count / len(self.point_data), 5)))
                print(self.items)

            # inter points reach the expected value
            if self.count > self.size:
                break
           
			self.items -= 1

        return self.best_model


def data_generator(contour):
    x_data = []
    y_data = []
    for point in contour:
        x_data.append(point[0])
        y_data.append(point[1])
    
    return x_data, y_data


if __name__ == '__main__':
	# 这个是根据我的工程实际问题写的寻找椭圆轮廓点，你们可以根据自己实际来该
	    # 1.find ellipse edge line 
	    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		points_data = np.reshape(contours, (-1, 2))  # ellipse edge points set

	    # 2.Ransac fit ellipse param 
	    Ransac = RANSAC(data=points_data, items=999, threshold=1, P=.96, S=.56)
	    (X, Y), (LAxis, SAxis), Angle = Ransac.execute_ransac()
