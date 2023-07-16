import numpy
import pydicom
import torch.optim
import torchvision
from pylab import *
from PIL import Image
from torch import nn

from . import picture_data_prepare_with_sampling as dataset
from sklearn import metrics
import network_L3 as network
import os
import cv2
from picture_test import data_preprocess_base
from picture_test import data_preprocess_base2
import matplotlib.pyplot as plt

# print('===============================================model:model_train=================================================')

# 参数设置
PATH = 'C:/Users/Ruan/Desktop/project/medical_imaging_recognition_in_enterprise_training/back/nettest/data/labels.csv'
TEST_PATH = ''
is_train = False  # True-训练模型  False-测试模型
save_model_name = 'model/L1_model.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 训练参数设置
SIZE = 224  # 图像进入网络的大小
BATCH_SIZE = 32  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 100  # 迭代次数

# 模型参数
backbone = 'alexnet'
pretrained = False

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径

# change
try:
    os.mkdir('model')  # 新建文件夹
except FileExistsError:
    pass

# 加载数据
# print('----before-call-2----')
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train)
# print('----after-Call-2----')

# 定义模型、优化器、损失函数
model = network.initialize_model(backbone=backbone, pretrained=pretrained, NUM_CLASS=NUM_CLASS)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


# 训练模型
def train_alexnet(model):
    for epoch in range(EPOCHS):
        correct = total = 0.
        loss_list = []
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.train()
            output = model(batch_x)  # forward
            loss = criterion(output, batch_y)  # backward

            # 优化过程
            optimizer.zero_grad()  # 梯度归0
            loss.backward()  # 计算调整量  参数+调整量
            optimizer.step()

            # 输出训练结果
            loss_list.append(loss.item())
            _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            # print('>>>>>>>>>>>>>>>>>>>>>>>one loop<<<<<<<<<<<<<<<<<<<<<<<<')
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, EPOCHS, train_avg_loss, train_avg_acc))

    # 保存模型
    torch.save(model.state_dict(), save_model_name)
    print('Training finished!')


# 测试模型  infer
def test_alexnet(model_name):
    print('------ Testing Start ------')
    model.load_state_dict(torch.load(model_name), False)
    test_pred = []
    test_true = []

    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_loader:
            # add 1 row
            images, labels = test_x, test_y
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            print('images:')
            print(images)
            output = model(images)
            print(output)
            _, predicted = torch.max(output.data, 1)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    print('test_classification_report\n', test_classification_report)
    print('Accuracy of the network is: %.4f %%' % test_acc)
    return test_acc, images, test_true, test_pred


def run_predict(model_name, picture_path: str):
    with torch.no_grad():
        global model
        model.load_state_dict(torch.load(model_name), False)
        dcm_info = pydicom.read_file(picture_path)
        dcm = dcm_info.pixel_array
        # plt.imshow(dcm)
        # plt.show()
        # print(1)
        # print(dcm)
        dcm = data_preprocess_base(dcm, 224)
        images = np.array(dcm, 'f')

        print(images.shape)


        # plt.imshow(dcm)
        # plt.show()
        # print(images.shape)     images[0]

        #resize  6x6-->1,2,3,3
        # reshape         0,3,6,6
        images = images.transpose(2,0,1)
        dcm1 = torch.tensor(images.astype(float32)).unsqueeze(0)  # .resize(1, 3, 224, 224)

        print(dcm1.shape)

        # dcm=dcm.transpose(0,3,1,2)

        # transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
        #                                              torchvision.transforms.ToTensor()])

        # dcm_img = transforms(dcm_info)
        #
        # in_features = model.fc.in_features
        #
        # model.fc = nn.Sequential(nn.Linear(in_features, 4096),
        #                             nn.Linear(4096, 2))
        #
        # model = torch.load("best_model_yaopian.pth", map_location=torch.device("cpu"))  # 选择训练后得到的模型文件
        # # print(model)
        # dcm_img = torch.reshape(dcm_img, (1, 3, 64, 64))

        model.eval()
        output = model(dcm1)
        print(output)
        data_class = ['bad', 'good']
        _, predicted = torch.max(output.data, 1)
        print(predicted)
        print(data_class[int(output.argmax(1))])
        # plt.imshow(dcm_img)
        # plt.show()
        # dcm_img = array(data_preprocess_base(dcm_img, 224))
        # label = model(dcm_img)
        # print(label)


run = False

if __name__ == '__main__':
    if not run:
        if is_train:
            train_alexnet(model)
        else:
            test_alexnet(save_model_name)
    else:
        run_predict(save_model_name, './data/images/00163.dcm')

# 保存模型
    model_name = 'model/L3_resnet18_best_model.pkl'
    torch.save(model.state_dict(), 'C:/Users/Ruan/Desktop/project/medical_imaging_recognition_in_enterprise_training/back/nettest/model/L3_resnet18_best_model.pkl')

# 加载模型
    model.load_state_dict(torch.load(model_name), False)

# image=Image.open(image_path)
# print(image)
# transforms=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
#                                           torchvision.transforms.ToTensor()])
# image=transforms(image)
# print(image.shape)
#
# model_ft=torchvision.models.resnet18()      #需要使用训练时的相同模型
# # print(model_ft)
# in_features=model_ft.fc.in_features
# model_ft.fc=nn.Sequential(nn.Linear(in_features,36),
#                           nn.Linear(36,6))     #此处也要与训练模型一致
#
# model=torch.load("best_model_yaopian.pth",map_location=torch.device("cpu")) #选择训练后得到的模型文件
# # print(model)
# image=torch.reshape(image,(1,3,64,64))
#
# model.eval()
# with torch.no_grad():
#     output=model(image)
