from collections import Counter

import numpy as np
import pydicom
import torch
from numpy import array
from torch.utils import data
from torchvision.transforms import transforms

import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file)

# 将当前目录添加到 sys.path 中
sys.path.append(current_dir)

# import model_train
from . import model_train
import network_L3 as network

from picture_test import data_preprocess_base


def test_alexnet(model_name, picture_path=None):
    print('------ Testing Start ------')
    model = network.initialize_model(backbone=model_train.backbone, pretrained=model_train.pretrained,
                                     NUM_CLASS=model_train.NUM_CLASS)
    model.load_state_dict(torch.load(model_name), False)
    test_pred = []
    test_true = []

    with torch.no_grad():
        model.eval()
        predict_loader = get_dataset(224, 1, picture_path)
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
            print(predicted)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    # images = predict_loader.dataset.train_img
    # test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    # test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    # print('test_classification_report\n', test_classification_report)
    # print('Accuracy of the network is: %.4f %%' % test_acc)

    return images, test_true, test_pred


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
        images = array([data_preprocess_base(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])
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

    # 定义train_loader
    predict_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 180), expand=False),
        transforms.ToTensor()])

    predict_set = PredictSet(predict_img, predict_label, predict_data_transform)
    predict_loader = data.DataLoader(dataset=predict_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return predict_loader


if __name__ == '__main__':
    i = 0
    while i < 800:
        i += 1
        path_base = './data/images/00'
        path_last = '.dcm'
        path_num = i.__str__()
        if i < 10:
            path_num = '00' + path_num
        elif i < 100:
            path_num = '0' + path_num
        path = path_base + path_num + path_last
        test_alexnet('model/L1_model.pkl', path)
