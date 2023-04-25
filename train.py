import torch               #pytorch
import torch.nn as nn      #pytorch
from my_flowers.dataset import val_dataset,train_dataset  #导入训练集和测试集的读取函数
from my_flowers.model import Classifies_model, Vgg16_Classifies_model  #导入分类模型
from my_flowers.MOCO_model import MOCO_Classifies_model
from my_flowers.images_arg import images_transform
import itertools
#%%

def train(val_batch_size = 200, train_batch_size = 300, model = MOCO_Classifies_model, is_load=False):
    test_data, long_val = val_dataset(batchsize=val_batch_size,is_load=is_load)  # 加载测试集和测试集长度long_val
    for i, (img_val, label_val) in enumerate(test_data):  # 读取测试集的图片img_val和类别label_val
        break
    img_val, label_val = img_val.cuda(), label_val.cuda()
    train_data, long = train_dataset(batchsize=train_batch_size,is_load=is_load)  # 加载训练集和训练集长度long
    train_data = itertools.cycle(train_data)  # 将训练集设置为循环模式以便训练的时候重复使用
    epoch = 80  # 训练轮数

    moco_class_model = model()
    moco_class_model = torch.nn.DataParallel(moco_class_model, device_ids=[0])
    moco_class_model = moco_class_model.cuda()
    opt = torch.optim.Adam(moco_class_model.parameters(), lr=0.0005)  # 定义一个优化器最小化目标函数（损失函数）
    loss = nn.CrossEntropyLoss().cuda()  # 分类交叉熵损失函数

    """用未训练的模型预测一轮测试集作为最佳score"""
    val_predict = moco_class_model(img_val)
    val_predict = torch.argmax(val_predict, 1)
    val_predict = val_predict.cuda()
    score_best = torch.sum(val_predict == label_val) / len(val_predict)  # 初始化最佳score
    val = []
    LOSS = []
    for i, (img, label_real) in enumerate(train_data):  # 读取测试集
        """
        img: 图片
        label_real: 花朵类别
        每张图片对应着一个类别，一共四个类别
        """
        if i > epoch * long:  # 当训练次数达到epoch时，结束训练
            break
        img, label_real = img.cuda(), label_real.cuda()
        label_real = label_real.squeeze(1)
        label_predict = moco_class_model(img)  # 用分类模型预测花朵类别label_predict，每张图片预测四个类别，数值最大的类别为最终预测类别
        function_loss = loss(label_predict, label_real)  # 交叉熵损失函数，将预测的花朵类别和真正的类别做交叉熵比较，
        #                                                   训练的目的就是为了让交叉熵越小越好，交叉熵越小，
        #                                                   代表预测的类别和真实类别越接近

        opt.zero_grad()  # 让优化器梯度归零，以便于更新损失函数
        function_loss.backward()  # 损失函数反向传播来更新分类网络参数，网络参数更新方向是损失函数减小的方向
        opt.step()
        """对三次随机数据增强后的数据集进行训练"""
        for j in range(2):
            img = img.cpu()
            img = images_transform(img)
            img = img.cuda()
            label_predict = moco_class_model(img)  # 用分类模型预测花朵类别label_predict，每张图片预测四个类别，数值最大的类别为最终预测类别
            function_loss = loss(label_predict, label_real)  # 交叉熵损失函数，将预测的花朵类别和真正的类别做交叉熵比较，
            #                                                   训练的目的就是为了让交叉熵越小越好，交叉熵越小，
            #                                                   代表预测的类别和真实类别越接近
            opt.zero_grad()  # 让优化器梯度归零，以便于更新损失函数
            function_loss.backward()  # 损失函数反向传播来更新分类网络参数，网络参数更新方向是损失函数减小的方向
            opt.step()

        if i % 3 == 0:  # 每训练10次就显示一次结果

            """计算模型预测 测试集 的准确度score"""
            val_predict = moco_class_model(img_val)  # 把训练得到的模型来预测测试集图片的类别，测试集的图片是模型从未见过的，
            #                                   用模型从未见过的图片测试模型是否真的学习到了花朵的特征
            val_predict = torch.argmax(val_predict, 1)  # 因为每张图片预测四个类别的概率值，概率值最大的类别为模型预测的类别，
            #                                                  这里的argmax函数就是为了得到模型预测出的四个类别概率中最大概率的类别
            score = torch.sum(val_predict == label_val) / len(val_predict)  # sum函数计算预测类别和真实类别相同的个数，
            #                                                               len(val_predict)是测试集图片的总数，二者相除得到准确率
            #                                                               测试集的准确度比训练集的准确度低，因为测试集是模型从未见过的图片
            if score > score_best:  # 如果这轮训练的测试结果比以往任何一次都要好
                torch.save(moco_class_model.state_dict(), '/mnt/my_flowers/resnet_best_model.pt')  # 保存这个分类网络的参数
                score_best = score  # 此时的score覆盖最佳score
            print(f"val_score:{score}%",epoch)  # 显示测试集的预测score
            print("loss:", function_loss)  # 显示全局损失函数
            val.append(score)
            LOSS.append(function_loss.item())
    return val,LOSS



if __name__ == "__main__":
    val,Loss = train()
