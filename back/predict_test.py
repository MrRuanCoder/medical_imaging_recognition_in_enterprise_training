from collections import Counter

import numpy as np
import pydicom
import torch
import torch.nn as nn
from numpy import array
from torch.utils import data
from torchvision.transforms import transforms
import datetime
import network_L3 as network
from picture_test import data_preprocess_enhanced
from PIL import Image

import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file)

# 将当前目录添加到 sys.path 中
sys.path.append(current_dir)

import model_train

import network_L3 as network

from picture_test import data_preprocess_base
from picture_test import data_preprocess_enhanced

def predict_(picture_path: list, model_path: str):
    # picture_path = picture_path[0]
    with torch.no_grad():
        model = network.initialize_model(backbone='resnet34', pretrained=False,
                                         NUM_CLASS=2)
        model.load_state_dict(torch.load(model_path), False)
        model.eval()
        images = []
        tensor_list = []
        predict_results = []
        for p_path in picture_path:
            dcm = pydicom.read_file(p_path).pixel_array
            image = data_preprocess_enhanced(dcm, 224)
            images.append(image)
        images = np.array(images)
        for img in images:
            img = Image.fromarray(np.uint8(img))  # narray->PIL
            predict_data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 180), expand=False),
                transforms.ToTensor()])
            img = predict_data_transform(img)
            tensor_list.append(img)
            # print(img)
        inputs = torch.stack(tensor_list)
        output = model(inputs)
        # print(output)
        _, predict_result = torch.max(output.data, 1)
    return predict_result.tolist()

def predict_enhanced(picture_path: str, model_path: str):
    picture_path = [picture_path]
    
    # Check if CUDA (GPU) is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model = network.initialize_model(backbone='resnet34', pretrained=False, NUM_CLASS=2)
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load model on the same device
        model.eval()

        images = []
        for p_path in picture_path:
            dcm = pydicom.read_file(p_path).pixel_array
            image = data_preprocess_enhanced(dcm, 224)
            images.append(image)
        images = np.array(images)

        # Data preprocessing using torchvision transforms
        predict_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180), expand=False),
            transforms.ToTensor()
        ])
        
        tensor_list = [predict_data_transform(Image.fromarray(np.uint8(img))) for img in images]
        inputs = torch.stack(tensor_list).to(device)  # Move inputs to the same device as the model

        output = model(inputs)
        _, predict_result = torch.max(output.data, 1)

    return predict_result



def test_alexnet(model_name, picture_path=None):
    print('------ Testing Start ------')
    # print("1")
    model = network.initialize_model(backbone=model_train.backbone, pretrained=model_train.pretrained,
                                     NUM_CLASS=model_train.NUM_CLASS)
    model.load_state_dict(torch.load(model_name), False)
    test_pred = []
    test_true = []
    # print('2')
    with torch.no_grad():
        print('1')
        model.eval()
        # print("2.1")
        predict_loader = get_dataset(224, 1, picture_path)
        print("2")
        starttime = datetime.datetime.now()
        # print(predict_loader)
        for test_x, test_y in predict_loader:
            # add 1 row
            print('2.1') 
            endtime = datetime.datetime.now()
            print(endtime-starttime)
            images, labels = test_x, test_y
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            # print('images:')
            # print(images)
            print('2.2')
            output = model(images)
            # print(output)
            _, predicted = torch.max(output.data, 1)
            print('2.3')
            print(predicted)                        
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))
            print('2.4')

    # images = predict_loader.dataset.train_img
    # test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    # test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    # print('test_classification_report\n', test_classification_report)
    # print('Accuracy of the network is: %.4f %%' % test_acc)

    return images, test_true, test_pred

def output_alexnet(model_name, picture_path=None):
    print('------ Testing Start ------')
    print("1")
    model = network.initialize_model(backbone='resnet34', pretrained=False,
                                     NUM_CLASS=2)
    model.load_state_dict(torch.load(model_name), False)
    test_pred = []
    test_true = []
    print("2")
     
    with torch.no_grad():
        model.eval()
        print("2.1")
        
        # infer
        # output = model(images)


        predict_loader = get_dataset(224, 1, picture_path)
        print("2.2")
        for test_x, test_y in predict_loader:
            # add 1 row
            images, labels = test_x, test_y
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            # print('images:')
            # print(images)
            output = model(images)
            # print(output)
            _, predicted = torch.max(output.data, 1)
#            print(predicted)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    # images = predict_loader.dataset.train_img
    # test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    # test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    # print('test_classification_report\n', test_classification_report)
    # print('Accuracy of the network is: %.4f %%' % test_acc)

    return predicted

def data_load(size, picture=None, picture_mode: str = 'path'):
    dicomlist = []  # 图像地址
    labels = []  # 图像标签
    train_img = []  # 训练集图像
    train_label = []  # 训练集标签
    val_img = []  # 验证集图像
    val_label = []  # 验证集标签

    # (1)读取数据：images图像矩阵，labels标签
    if picture_mode == 'path':
        dicomlist.append(picture)
        labels.append('0')
        # 读取图像矩阵
        images = array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])
    else:
        images = None

    # (2)划分数据集
    # if is_train or test_path == '':  # 训练模式或测试模式没有单独csv
    #     print('----Training Mode----') if is_train else print('----Testing mode----')
    #     # 划分数据集：训练集、验证集、测试集
    #     images, labels = sklearn.utils.shuffle(images, labels, random_state=random_seed)  # images = n*224*224
    #     train_val_img, test_img, train_val_label, test_label = train_test_split(images, labels, test_size=ratio,
    #                                                                             stratify=labels,
    #                                                                             random_state=random_seed)
    #
    #     train_img, val_img, train_label, val_label = train_test_split(train_val_img, train_val_label,
    #                                                                   test_size=ratio, stratify=train_val_label,
    #                                                                   random_state=random_seed)
    #
    #     print('Dataset: %s, labels=%s' % (images.shape, sorted(Counter(labels).items())))
    #     print('Training set: %s, labels=%s' % (train_img.shape, sorted(Counter(train_label).items())))
    #     print('Val set: %s, labels=%s' % (val_img.shape, sorted(Counter(val_label).items())))
    #     print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

    # 测试模式
    # print('----Testing Mode----')
    predict_img = images
    predict_label = labels
    # print('Test set: %s, labels=%s' % (predict_img.shape, sorted(Counter(predict_label).items())))

    return predict_img, predict_label


class PredictSet(data.Dataset):
    def __init__(self, train_img, train_label, train_data_transform=None):
        super(PredictSet, self).__init__()
        self.train_img = train_img
        self.train_label = train_label
        self.train_data_transform = train_data_transform

    def __getitem__(self, index):
        img = self.train_img[index]
        target = int(self.train_label[index])
        if self.train_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))  # narray->PIL
            img = self.train_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.train_img)


def get_dataset(size, batch_size, picture=None, picture_mode='path'):
    
    predict_img, predict_label = data_load(size, picture, picture_mode)
    # starttime = datetime.datetime.now()
    # 定义train_loader
    predict_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 180), expand=False),
        transforms.ToTensor()])
    
    predict_set = PredictSet(predict_img, predict_label, predict_data_transform)
    predict_loader = data.DataLoader(dataset=predict_set, batch_size=batch_size, shuffle=False, num_workers=4
                                    #  , pin_memory=True
                                     )
    # endtime = datetime.datetime.now()   
    # print(endtime - starttime)
    
    
    return predict_loader


if __name__ == '__main__':
    i = 0
    while i < 500:
        i += 1
        path_base = './data/images/00'
        path_last = '.dcm'
        path_num = i.__str__()
        if i < 10:
            path_num = '00' + path_num
        elif i < 100:
            path_num = '0' + path_num
        path = path_base + path_num + path_last
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(predict(path, 'model/resnet34-picture-enhance.pkl'))
        print('################################')
