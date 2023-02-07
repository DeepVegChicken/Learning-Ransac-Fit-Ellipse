# Learning-Ransac-Fit-Ellipse
Python 使用Ransac拟合椭圆

Note：

主要看RANSAC.py文件

里面有些东西要删除的，我从项目中摘出来的，抱歉

def main()是主函数入口

main函数中有注释部分是对应的拟合函数部分

![131231](https://user-images.githubusercontent.com/36610446/217200519-4a5ca406-013e-4b97-994f-73c87dbab00c.PNG)

RANSAC(img=(img_, _img_), data=points_data, n=8, max_refines=3, max_inliers=90.0, threshold=1.0, show=False)这句中的输入参数介绍在RANSAC类中的__init__函数中

另外，RANSAC类中，执行过程如下：

![21313113123131](https://user-images.githubusercontent.com/36610446/217201593-6ebdea59-9daa-40ab-b058-6f13731e205f.PNG)


