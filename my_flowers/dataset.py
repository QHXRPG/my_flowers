import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image



def train_dataset(batchsize=100,is_load = True):
    # 设置文件夹路径和标签
    if is_load:
        folder_paths = ['/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/train/dandelion',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/train/roses',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/train/sunflowers',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/train/tulips']
    else:
        folder_paths = ['/mnt/flowers_4/train/dandelion',
                        '/mnt/flowers_4/train/roses',
                        '/mnt/flowers_4/train/sunflowers',
                        '/mnt/flowers_4/train/tulips']
    labels = np.array([0, 1, 2, 3])
    class_labels = []
    img_all = []
    labels_all = []
    # 初始化数组
    num_images1 = sum([len(files) for r1, d1, files in os.walk(folder_paths[0])])
    num_images2 = sum([len(files) for r2, d2, files in os.walk(folder_paths[1])])
    num_images3 = sum([len(files) for r3, d3, files in os.walk(folder_paths[2])])
    num_images4 = sum([len(files) for r4, d4, files in os.walk(folder_paths[3])])
    num_images = num_images1 + num_images2 +num_images3 + num_images4
    images = np.zeros((num_images, 3, 64, 64),dtype=np.float32)
    count = 0

    # 删除隐藏文件
    for folder_path, label in zip(folder_paths, labels):
        for filename in os.listdir(folder_path):
            if filename == '.DS_Store':
                os.remove(folder_path + '/' + filename)

    # 读取每个文件夹中的图像并存储为numpy数组
    for folder_path, label in zip(folder_paths, labels):
        for filename in os.listdir(folder_path):
            if folder_path.split('/')[-1] == 'dandelion':
                class_labels.append(0)
            elif folder_path.split('/')[-1] == 'roses':
                class_labels.append(1)
            elif folder_path.split('/')[-1] == 'sunflowers':
                class_labels.append(2)
            elif folder_path.split('/')[-1] == 'tulips':
                class_labels.append(3)
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = np.array(image)
            images[count] = np.transpose(image, (2, 0, 1))
            count += 1
    shuffle = torch.randperm(num_images)
    images = torch.from_numpy(images)
    images = images[shuffle]   #打乱顺序
    class_labels = np.array(class_labels)
    class_labels = torch.from_numpy(class_labels)
    class_labels = class_labels.unsqueeze(1)
    class_labels = class_labels[shuffle]  #打乱顺序
    for i in range(int(round(len(class_labels) / batchsize, 0))):
        if (i + 1) * batchsize < len(class_labels):
            img_all.append(images[i * batchsize: (i + 1) * batchsize])
            labels_all.append(class_labels[i * batchsize: (i + 1) * batchsize])
        else:
            img_all.append(images[i * batchsize:])
            labels_all.append(class_labels[i * batchsize:])
    return zip(img_all, labels_all), len(list(zip(img_all, labels_all)))

def val_dataset(batchsize=100, is_load = True):
    # 设置文件夹路径和标签
    if is_load == False:
        folder_paths = ['/mnt/flowers_4/val/dandelion',
                        '/mnt/flowers_4/val/roses',
                        '/mnt/flowers_4/val/sunflowers',
                        '/mnt/flowers_4/val/tulips']
    else:
        folder_paths = ['/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/val/dandelion',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/val/roses',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/val/sunflowers',
                        '/Users/qiuhaoxuan/Desktop/数据集/myflowe/data/val/tulips']
    labels = np.array([0, 1, 2, 3])
    class_labels = []
    img_all = []
    labels_all = []
    # 初始化数组
    num_images1 = sum([len(files) for r1, d1, files in os.walk(folder_paths[0])])
    num_images2 = sum([len(files) for r2, d2, files in os.walk(folder_paths[1])])
    num_images3 = sum([len(files) for r3, d3, files in os.walk(folder_paths[2])])
    num_images4 = sum([len(files) for r4, d4, files in os.walk(folder_paths[3])])
    num_images = num_images1 + num_images2 +num_images3 + num_images4
    images = np.zeros((num_images, 3, 64, 64),dtype=np.float32)
    count = 0

    # 删除隐藏文件
    for folder_path, label in zip(folder_paths, labels):
        for filename in os.listdir(folder_path):
            if filename == '.DS_Store':
                os.remove(folder_path + '/' + filename)

    # 读取每个文件夹中的图像并存储为numpy数组
    for folder_path, label in zip(folder_paths, labels):
        for filename in os.listdir(folder_path):
            if folder_path.split('/')[-1] == 'dandelion':
                class_labels.append(0)
            elif folder_path.split('/')[-1] == 'roses':
                class_labels.append(1)
            elif folder_path.split('/')[-1] == 'sunflowers':
                class_labels.append(2)
            elif folder_path.split('/')[-1] == 'tulips':
                class_labels.append(3)
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = np.array(image)
            images[count] = np.transpose(image, (2, 0, 1))
            count += 1
    images = torch.from_numpy(images)
    class_labels = np.array(class_labels)
    class_labels = torch.from_numpy(class_labels)
    for i in range(int(round(len(class_labels) / batchsize, 0))):
        if (i + 1) * batchsize < len(class_labels):
            img_all.append(images[i * batchsize: (i + 1) * batchsize])
            labels_all.append(class_labels[i * batchsize: (i + 1) * batchsize])
        else:
            img_all.append(images[i * batchsize:])
            labels_all.append(class_labels[i * batchsize:])
    return zip(img_all, labels_all), len(list(zip(img_all, labels_all)))
