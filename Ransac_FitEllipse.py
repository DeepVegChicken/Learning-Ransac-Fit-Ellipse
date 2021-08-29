# *_* coding:utf-8 _*_
# 开发团队: 喵里的猫
# 开发人员: XFT
# 开发时间: 2021/8/29 12:29
# 文件名称: Ransac_FitEllipse.py
# 开发工具: PyCharm


import os
import cv2
import math
import time
import random
import numpy as np
import matplotlib
from random import shuffle
import matplotlib.pylab as plt


class RANSAC:
    def __init__(self, img, data, n, max_refines, max_inliers, threshold, show):
        self.img, self.orig_img = img  # 分割后图片、label图片
        self.point_data = data  # 椭圆轮廓点集
        self.length = len(self.point_data)  # 椭圆轮廓点集长度
        self.show = show

        self.n = n  # 随机选取n个点
        self.max_items = 999  # 迭代次数
        self.inliers_num = 0  # 内点计数器
        self.max_refines = max_refines  # 最大重迭代次数
        self.max_perc_inliers = max_inliers  # 最大内点比例

        self.error_threshold = threshold  # 模型误差容忍阀值
        self.best_model = ((0, 0), (1e-6, 1e-6), 0)  # 椭圆模型存储器

    def random_sampling(self, n):
        """随机取n个数据点"""
        select_point = []
        all_point = self.point_data

        k, m = divmod(len(all_point), n)
        list_split = [all_point[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]
        select_patch = random.sample(list_split, n // 2)
        for list_ in select_patch:
            temp = random.sample(list(list_), n // 2 // 2)
            select_point.append(temp[0])
            select_point.append(temp[1])

        return np.array(select_point)
        # return np.asarray(random.sample(list(self.point_data), n))

    def Geometric2Conic(self, ellipse):
        """ Calculate the elliptic equation coefficients """
        # Ax ^ 2 + Bxy + Cy ^ 2 + Dx + Ey + F
        (x0, y0), (bb, aa), phi_b_deg = ellipse

        # Semimajor and semiminor axes
        a, b = aa / 2, bb / 2
        # Convert phi_b from deg to rad
        phi_b_rad = phi_b_deg * np.pi / 180.0
        # Major axis unit vector
        ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

        # Useful intermediates
        a2 = a * a
        b2 = b * b

        # Conic parameters
        if a2 > 0 and b2 > 0:
            A = ax * ax / a2 + ay * ay / b2
            B = 2 * ax * ay / a2 - 2 * ax * ay / b2
            C = ay * ay / a2 + ax * ax / b2
            D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
            E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
            F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + \
                (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1
        else:
            # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
            A, B, C, D, E, F = (1, 0, 1, 0, 0, -1e-6)

        # Compose conic parameter array
        conic = np.array((A, B, C, D, E, F))

        return conic

    def ConicFunctions(self, points, ellipse):
        """
        Calculate various conic quadratic curve support functions
        Parameters
        ----
        points : n x 2 array of floats
        ellipse : tuple of tuples

        Returns
        ----
        distance : array of floats
        grad : array of floats
        absgrad : array of floats
        normgrad : array of floats
        """
        # Convert from geometric to conic ellipse parameters
        conic = self.Geometric2Conic(ellipse)

        # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
        C = np.array(conic)

        # Extract vectors of x and y values
        x, y = points[:, 0], points[:, 1]

        # Construct polynomial array (6 x n)
        X = np.array((x * x, x * y, y * y, x, y, np.ones_like(x)))

        # Calculate Q/distance for all points (1 x n)
        distance = C.dot(X)

        # Construct conic gradient coefficients vector (2 x 3)
        Cg = np.array(((2 * C[0], C[1], C[3]), (C[1], 2 * C[2], C[4])))

        # Construct polynomial array (3 x n)
        Xg = np.array((x, y, np.ones_like(x)))

        # Gradient array (2 x n)
        grad = Cg.dot(Xg)

        # Normalize gradient -> unit gradient vector
        absgrad = np.sqrt(np.sqrt(grad[0, :] ** 2 + grad[1, :] ** 2))
        normgrad = grad / absgrad

        return distance, grad, absgrad, normgrad

    def EllipseError(self, points, ellipse):
        # Calculate algebraic distances and gradients of all points from fitted ellipse
        distance, grad, absgrad, normgrad = self.ConicFunctions(points, ellipse)

        # eps = 2.220446049250313e-16
        eps = np.finfo(distance.dtype).eps
        absgrad = np.maximum(absgrad, eps)

        # Calculate error from distance and gradient
        err = distance / absgrad
        return err

    def EllipseNormError(self, points, ellipse):
        (x0, y0), (bb, aa), phi_b_deg = ellipse

        # Semiminor axis
        b = bb / 2
        # Convert phi_b from deg to rad
        phi_b_rad = phi_b_deg * np.pi / 180.0
        # Minor axis vector
        bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)

        # Point one pixel out from ellipse on minor axis
        p1 = np.array((x0 + (b + 1) * bx, y0 + (b + 1) * by)).reshape(1, 2)

        # Error at this point
        err_p1 = self.EllipseError(p1, ellipse)

        # Errors at provided points
        err_pnts = self.EllipseError(points, ellipse)

        return err_pnts / err_p1

    def OverlayRANSACFit(self, overlay, outlier_points, inlier_points, best_ellipse):
        """ Ransac IMG """
        # outlier_points in green
        for col, row in outlier_points:
            overlay[row, col] = [0, 255, 0]
        # inlier_points in red
        for col, row in inlier_points:
            overlay[row, col] = [0, 0, 255]

        """ Seg_Orign """
        # all points in green
        for col, row in self.point_data:
            self.img[row, col] = [0, 255, 0]
        # inlier fitted ellipse in red
        cv2.ellipse(self.img, best_ellipse, (0, 0, 255), 1)

        """ Label_Orign """
        # all points in green
        for col, row in self.point_data:
            self.orig_img[row, col] = [0, 255, 0]
        # # inlier fitted ellipse in red
        cv2.ellipse(self.orig_img, best_ellipse, (0, 0, 255), 1)

    def execute_ransac(self):
        count = 0
        while count < int(self.max_items) + 1:
            count += 1
            # 1.select n points at random
            select_points = self.random_sampling(self.n)

            # 2.fitting selected ellipse points
            ellipse = cv2.fitEllipse(select_points)

            # 3.Refine inliers iteratively
            for refine in range(self.max_refines):
                # Calculate normalized errors for all points
                norm_err = self.EllipseNormError(self.point_data, ellipse)
                # Identify inliers and outlier
                inliers = np.nonzero(norm_err ** 2 < self.error_threshold)[0]
                outlier = np.nonzero(norm_err ** 2 >= self.error_threshold)[0]
                # Update inliers and outlier set
                inlier_points = self.point_data[inliers]
                outlier_points = self.point_data[outlier]
                if inliers.size < 5:
                    break
                # Fit ellipse to refined inliers set
                ellipse = cv2.fitEllipse(np.asarray(inlier_points))

            # 4.Count inliers ratio
            inliers_lenght = inliers.size
            perc_inliers = inliers_lenght / self.length
            if inliers_lenght > self.inliers_num:
                # 5.Update param
                self.best_model = ellipse
                self.inliers_num = inliers_lenght
                # 6.Judge whether the current inliers is greater than the maximum inliers
                if perc_inliers * 100.0 > self.max_perc_inliers:
                    break
                if perc_inliers > 0.08:
                    self.max_items = count + math.log(1 - 0.99) / math.log(1 - pow(perc_inliers, self.n))

        # Update fitted image and display
        if self.show:
            overlay = np.zeros((512, 512, 3), np.uint8)
            self.OverlayRANSACFit(overlay, outlier_points, inlier_points, self.best_model)
            cv2.imshow('Ransac', overlay)
            cv2.imshow('Seg_Orign', self.img)
            cv2.imshow('Label_Orign', self.orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.best_model


def main():
    OrignDir = 'DataSet/TrainSet/Label/'
    PictureDir = 'DataSet/TestResults/oval/'
    OvalDir = 'DataSet/Model_Data/test.txt'
    Test_Results = 'DataSet/Model_Data/ScaleRecord.txt'
    with open(OvalDir) as f:
        lines = f.read().splitlines()
    with open(Test_Results) as f:
        Test_lines = f.read().splitlines()

    f = open('DataSet/TestResults/TestRecord.txt', 'w')
    for line in lines:
        Name, name = line.split(',')[0], line.split(',')[1]
        img = cv2.imread(PictureDir + name, 0)  # 单通道分割图
        img_ = cv2.imread(PictureDir + name)  # 三通道分割图
        _img_ = cv2.imread(OrignDir + name)  # 对应的Label图

        # Search magnification factor
        for lin in Test_lines:
            _Name, _name = lin.split(',')[0], lin.split(',')[1]
            if name == _name:
                scale_x, scale_y = [(float(i)) for i in (lin.split(',')[2:4])]
                break
        line_true = [(float(i)) for i in (line.split(',')[2:])]
        x, y, axil_x, axil_y, theta = line_true[:5]
        xmin, ymin, xmax, ymax = line_true[5:]
        theta = (math.pi + theta) if theta < 0 else theta
        theta = theta * 180 / math.pi

        # 这部分代码为拟合代码，只需将对应的参数送入即可
        # 其中img_为待拟合的图片，_img_可以根据需要进行删改
        # --------------------------------------------------------------------------------------------------------------
        # 1.find the max ellipse edge line
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        maxlength_contours = max(contours, key=len)  # select max_length edge line

        # 2.Ransac fit ellipse param
        points_data = np.reshape(maxlength_contours, (-1, 2))  # ellipse edge points set
        Ransac = RANSAC(img=(img_, _img_), data=points_data, n=8, max_refines=3, max_inliers=90.0, threshold=1.0, show=False)
        (X, Y), (LAxis, SAxis), Angle = Ransac.execute_ransac()
        # --------------------------------------------------------------------------------------------------------------

        # calculate bias
        centre_bias_x = abs((X / scale_x) - (x - xmin))
        centre_bias_y = abs((Y / scale_y) - (y - ymin))
        axil_bias_long = abs((LAxis / 2 / scale_x) - axil_x)
        axil_bias_short = abs((SAxis / 2 / scale_y) - axil_y)
        Angle = (Angle - 90) if Angle > 90 else (Angle + 90)
        theta_bias = abs(Angle - theta)

        # record bias
        f.write(Name + ',' + name + ',' +
                str(centre_bias_x) + ',' + str(centre_bias_y) + ',' +
                str(axil_bias_long) + ',' + str(axil_bias_short) + ',' +
                str(theta_bias) + '\n')

        print('Finshed ', Name, name)
    f.close()
    print('Well finshed')


if __name__ == '__main__':
    main()
