from collections import Counter
import os
import pydicom
import sklearn.utils
import torch.utils.data as data
from pylab import *
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

from picture_test import data_preprocess_base

PATH = 'C:/Users/Ruan/Desktop/project/medical_imaging_recognition_in_enterprise_training/back/nettest/data/labels.csv'  # 数据集路径
TEST_PATH = ''  # 测试集路径
is_train = False  # True-训练模型  False-测试模型
SIZE = 224  # 图像进入网络的大小
BATCH_SIZE = 32  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 50  # 迭代次数
random_seed = 321  # 随机种子
ratio = 0.1  # 验证集、测试集比例


# 上采样
def over_sampling(train_img, train_label):
    from imblearn.over_sampling import RandomOverSampler
    rus = RandomOverSampler(random_state=random_seed)
    nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled


def data_load(path, test_path, size, is_train):
    dicomlist = []  # 图像地址
    labels = []  # 图像标签
    train_img = []  # 训练集图像
    train_label = []  # 训练集标签
    val_img = []  # 验证集图像
    val_label = []  # 验证集标签

    # (1)读取数据：images图像矩阵，labels标签
    f = open(path, "r+") if test_path == '' else open(test_path, "r+")
    for line in f.readlines():
        img_path = os.path.join('', line.strip().split(',')[0])  # 图像地址
        dicomlist.append(img_path)
        label = line.strip().split(',')[1]  # 图像标签
        label = '0' if label == 'good' else '1'
        labels.append(label)
    labels = np.array(labels)  # 图像标签 n*1
    # 读取图像矩阵
    images = array([data_preprocess_base(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])
    f.close()

    # (2)划分数据集
    if is_train or test_path == '':  # 训练模式或测试模式没有单独csv
        print('----Training Mode----') if is_train else print('----Testing mode----')
        # 划分数据集：训练集、验证集、测试集
        images, labels = over_sampling(images, labels)
        images, labels = sklearn.utils.shuffle(images, labels, random_state=random_seed)  # images = n*224*224
        train_val_img, test_img, train_val_label, test_label = train_test_split(images, labels, test_size=ratio,
                                                                                stratify=labels,
                                                                                random_state=random_seed)

        train_img, val_img, train_label, val_label = train_test_split(train_val_img, train_val_label,
                                                                      test_size=ratio, stratify=train_val_label,
                                                                      random_state=random_seed)

        print('Dataset: %s, labels=%s' % (images.shape, sorted(Counter(labels).items())))
        print('Training set: %s, labels=%s' % (train_img.shape, sorted(Counter(train_label).items())))
        print('Val set: %s, labels=%s' % (val_img.shape, sorted(Counter(val_label).items())))
        print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

    else:  # 测试模式
        print('----Testing Mode----')
        test_img = images
        test_label = labels
        print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

    return train_img, train_label, val_img, val_label, test_img, test_label


train_img, train_label, val_img, val_label, test_img, test_label = data_load(PATH, TEST_PATH, SIZE, is_train=is_train)


class TrainDataset(data.Dataset):
    def __init__(self, train_img, train_label, train_data_transform=None):
        super(TrainDataset, self).__init__()
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


class ValDataset(data.Dataset):
    def __init__(self, val_img, val_label, val_data_transform):
        super(ValDataset, self).__init__()
        self.val_img = val_img
        self.val_label = val_label
        self.val_data_transform = val_data_transform

    def __getitem__(self, index):
        img = self.val_img[index]
        target = int(self.val_label[index])
        if self.val_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))  # narray->PIL
            img = self.val_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.val_img)


class TestDataset(data.Dataset):
    def __init__(self, test_img, test_label, test_data_transform):
        super(TestDataset, self).__init__()
        self.test_img = test_img
        self.test_label = test_label
        self.test_data_transform = test_data_transform

    def __getitem__(self, index):
        img = self.test_img[index]
        target = int(self.test_label[index])
        if self.test_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.test_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.test_img)


# -------------------------------#
#          加载数据集
# -------------------------------#
def get_dataset(path, test_path, size, batch_size, is_train, is_sampling=False):
    train_img, train_label, val_img, val_label, test_img, test_label = data_load(path, test_path, size, is_train)
    # is_sampling
    train_loader = []
    val_loader = []

    if is_train:
        # 定义train_loader
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180), expand=False),
            transforms.ToTensor()])

        train_set = TrainDataset(train_img, train_label, train_data_transform)
        train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)

        # 定义val_loader
        val_data_transform = transforms.Compose([
            transforms.ToTensor()])
        val_set = ValDataset(val_img, val_label, val_data_transform)
        val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义test_loader
    test_data_transform = transforms.Compose([
        transforms.ToTensor()])
    test_set = TestDataset(test_img, test_label, test_data_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader




# 下采样
def under_sampling(train_img, train_label):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_seed, replacement=False)
    nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled

#
# train_img, train_label, val_img, val_label, test_img, test_label = data_load(PATH, TEST_PATH, SIZE, is_train=is_train)
# print('方式1：无采样训练集正负样本分布情况')
# print(sorted(Counter(train_label).items()))
#
# print('方式2：上采样训练集正负样本分布情况')
# over_img, over_label = over_sampling(train_img, train_label)
# print(sorted(Counter(over_label).items()))
#
# print('方式3：下采样训练集正负样本分布情况')
# under_img, under_label = under_sampling(train_img, train_label)
# print(sorted(Counter(under_label).items()))
