import torch

from my_flowers.dataset import val_dataset,train_dataset
from my_flowers.model import Classifies_model,Vgg16_Classifies_model
from my_flowers.MOCO_model import MOCO_Classifies_model


def predict():
    test_data, long = val_dataset(batchsize=200)

    for i, (x, y) in enumerate(test_data):
        break
    x = x.cuda()
    y = y.cuda()
    model = Vgg16_Classifies_model()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    model.load_state_dict(torch.load("/mnt/my_flowers/vgg_model_best.pt"))
    print("测试集所有图片数：",len(x))
    predict_label = model(x)
    predict_label = torch.argmax(predict_label,1)
    score = torch.sum(predict_label == y)/len(x)
    print(f"预测准确度:{score*100}%")

if __name__ == '__main__':
    predict()