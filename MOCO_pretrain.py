#%%
import torch
import torch.nn as nn
import torchvision.models as models
import itertools
from my_flowers.dataset import val_dataset,train_dataset
from my_flowers.images_arg import images_transform
model1 = models.resnet50(pretrained=True)
model2 = models.resnet50(pretrained=True)



"""起码要用两块A4000训练"""
#%%

"""两个编码器"""
class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.l1 = nn.Sequential(*list(model1.children())[:-1])
        self.l2 = nn.Sequential(nn.Linear(2048,1000))

    def forward(self, x):
        x = self.l1(x)
        B, _, _, _ = x.shape
        output = x.view(B,-1)
        output = self.l2(output)
        return output


class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()
        self.l1 = nn.Sequential(*list(model2.children())[:-1])
        self.l2 = nn.Sequential(nn.Linear(2048,1000))

    def forward(self, x):
        x = self.l1(x)
        B, _, _, _ = x.shape
        output = x.view(B,-1)
        output = self.l2(output)
        return output

f_k = encoder1()
f_q = encoder2()
f_q = torch.nn.DataParallel(f_q, device_ids=[0, 1])
f_q = f_q.cuda()
f_k = torch.nn.DataParallel(f_k, device_ids=[0, 1])
f_k = f_k.cuda()



"""冲量更新函数"""
def moco_updata(f_k,f_q):
    m = 0.999
    for i, j in zip(f_k.parameters(), f_q.parameters()):
        i.data = i.data * m + j.data * (1. - m)


"""更新队列"""
class Queue(nn.Module):
    def __init__(self):
        super(Queue, self).__init__()
        self.length = 0
        self.queue = torch.rand(0)
    def updata(self,k):
        k_img = k.cpu()
        K, C = k_img.shape
        self.length = self.length + K
        self.queue = torch.cat([k_img,self.queue])
        if self.length >50000:
            self.queue = self.queue[:50000]
            self.length = 50000
        self.que = self.queue.transpose(1,0)
queue = Queue()
queue = queue
Moco_loss = []
if __name__ == "__main__":
    test_data, long_val = val_dataset(batchsize=200,is_load=False)  # 加载测试集和测试集长度long_val
    for i, (img_val, label_val) in enumerate(test_data):  # 读取测试集的图片img_val和类别label_val
        break
    train_data, long = train_dataset(batchsize=200,is_load=False)  # 加载训练集和训练集长度long
    train_loader = itertools.cycle(train_data)  # 将训练集设置为循环模式以便训练的时候重复使用
    epoch = 28  # 训练轮数
    T = 4
    opt = torch.optim.Adam(f_q.parameters(), lr=0.0002)  #定义一个优化器最小化目标函数（损失函数）
    loss = nn.CrossEntropyLoss().cuda()  #分类交叉熵损失函数
    """训练"""
    """初始化两个encoder网络"""
    for param in f_q.parameters():
        param.requires_grad = True    #设置f_q网络可以梯度下降用来训练
    for param in f_k.parameters():
        param.requires_grad = False   #冻结f_k网络，不用于梯度下降

    """初始化队列"""
    for j, (img, y) in enumerate(train_loader):
        break
    img_q = images_transform(img)
    img_k = images_transform(img)  #进行图像随机增强
    img_q,img_k = img_q.cuda(),img_k.cuda()
    k = f_k(img_q)
    queue.updata(k)   #更新队列

    for i,(img,y) in enumerate(train_loader):
        if i>epoch*long:
            break
        img_q = images_transform(img)
        img_k = images_transform(img)   #数据增强
        img_q, img_k = img_q.cuda(), img_k.cuda()
        q = f_q(img_q)
        k = f_k(img_k)  #正样本对
        k = k.detach()
        B,C = q.shape
        Q = queue.que   #负样本
        Q = Q.cuda()
        l_neg = q @ Q #(B,K)
        l_pos = (q.view(B,1,C) @ k.view(B,C,1)).view(B,-1) #(B,1)
        logits = torch.cat([l_pos,l_neg],dim=1)
        labels = torch.zeros(B,dtype=torch.long)
        logits, labels = logits.cuda(), labels.cuda()    #gpu
        l = loss(logits/T,labels)
        print("l:",l)
        Moco_loss.append(l.item())
        opt.zero_grad()
        l.backward(retain_graph=True) #更新f_q
        moco_updata(f_k,f_q)  #动量更新f_k
        opt.step()
        queue.updata(k)
    torch.save(f_q.state_dict(), '/mnt/f_q.pt')
    torch.save(f_k.state_dict(), '/mnt/f_k.pt')   #保存两个encoder


    #%%
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建一个示例数组
    x = np.arange(0, 393, 1)
    y = np.array(Moco_loss)

    # 设置图形大小和dpi
    fig = plt.figure(figsize=(10, 10), dpi=80)

    # 添加一个子图
    ax = fig.add_subplot(1, 1, 1)

    # 设置x轴和y轴的标签
    ax.set_xlabel('Epoch',fontsize=28)
    ax.set_ylabel('Loss',fontsize=28)

    # 设置x轴和y轴的范围
    ax.set_xlim([0, 393])
    ax.set_ylim([0, 8])

    # 添加标题
    ax.set_title('MOCO LOSS',fontsize=28)

    # 绘制折线图
    ax.plot(x, y, color='blue', linewidth=2.0, linestyle='-')

    # 显示图形
    plt.show()