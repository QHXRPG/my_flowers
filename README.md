# my_flowers
<img width="826" alt="image" src="https://user-images.githubusercontent.com/105186795/234267862-b9d062e9-8581-432d-9c2c-a5478509e4a0.png">




<img width="906" alt="image" src="https://user-images.githubusercontent.com/105186795/234268122-4e88a1bb-ad15-42a9-830e-c52fc03772d5.png">




<img width="909" alt="image" src="https://user-images.githubusercontent.com/105186795/234268256-20e34843-7f2c-4bb2-b1cf-b80923f5bfcd.png">




<img width="959" alt="image" src="https://user-images.githubusercontent.com/105186795/234268329-798a1257-b0e5-4b10-8408-9d45bd1a18fc.png">


该试验所用到的数据增强方法：
1.调整亮度：adjust_brightness()方法，通过调整图像的HSV空间中的V通道实现。
2.调整对比度：adjust_contrast()方法，通过将图像的像素值减去均值再乘以一个因子加上均值实现。
3.锐化：sharpen()方法，通过卷积图像和一个锐化核实现。
4.模糊：blur()方法，通过高斯滤波器实现。
5.调整颜色：adjust_color()方法，通过调整图像的HSV空间中的S通道实现。
6.旋转：rotate()方法，通过旋转矩阵实现。
7.水平翻转：flip_horizontal()方法，通过水平翻转实现。
8.垂直翻转：flip_vertical()方法，通过垂直翻转实现。
arg:固定数据增强：一个图像分别通过1-8种数据增强的方法生成8个经过数据增强的图像，
再与原图像一起作为训练集进行训练

arg+随机数据增强：一个图像随机通过两个或以上的数据增强方法生成3个经过数据增强的图像，再与原图像一起作为训练集进行训练。
 优点：1.增强训练集随机性和复杂度，降低过拟合风险。2减少数据集的数量，大大降低网络训练时间和拟合时间。
 
 <img width="678" alt="image" src="https://user-images.githubusercontent.com/105186795/234268576-21cd6ea9-c532-466f-82d0-391d67674566.png">

 
 
 
 <img width="601" alt="image" src="https://user-images.githubusercontent.com/105186795/234268614-7ad8b1dd-b845-4e07-88b1-e5da34375c49.png">
