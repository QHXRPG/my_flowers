import torch
import torch.nn as nn
import random
import numpy as np
import itertools
import cv2
from my_flowers.dataset import val_dataset,train_dataset  #导入训练集和测试集的读取函数

class ImageEnhancer(nn.Module):
    def __init__(self):
        super(ImageEnhancer, self).__init__()

    def adjust_brightness(self, img, factor=1.5):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, img, factor=2.0):
        mean = np.mean(img)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    def sharpen(self, img):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)

    def blur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    def adjust_color(self, img, factor=1.5):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def rotate(self, img, angle=45):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def flip_horizontal(self, img):
        return cv2.flip(img, 1)

    def flip_vertical(self, img):
        return cv2.flip(img, 0)

    def forward(self,img):
        ditk = {0:self.adjust_brightness, 1:self.adjust_contrast, 2:self.sharpen,3:self.blur, 4:self.adjust_color,
                5:self.rotate, 6:self.flip_horizontal,7:self.flip_vertical}
        step_1 = ditk[random.randint(0,4)]
        step_2 = ditk[random.randint(5,7)]
        output = step_2(step_1(img))
        return output

def images_transform(img:torch.Tensor):
    imageenhancer = ImageEnhancer()
    img = img.permute(0,2,3,1).contiguous()
    img = img.numpy()
    B,W,H,C = img.shape
    img_enhancers = np.zeros((B,W,H,C))
    for j in range(B):
        img_enhancers[j] = imageenhancer(img[j])
    img_enhancers = torch.from_numpy(img_enhancers)
    img_enhancers = img_enhancers.permute(0,3,1,2)
    img_enhancers = img_enhancers.float()
    return img_enhancers




if __name__ == '__main__':
    train_data, long = train_dataset(batchsize=60, is_load=True)  # 加载训练集和训练集长度long
    train_data = itertools.cycle(train_data)  # 将训练集设置为循环模式以便训练的时候重复使用
    for i, (img, label_real) in enumerate(train_data):  # 读取测试集
        break
    img_enhancers = images_transform(img)