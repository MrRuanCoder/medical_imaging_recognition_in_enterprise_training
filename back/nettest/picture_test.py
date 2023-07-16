import pydicom
import matplotlib.pyplot as plt
from pylab import *
import pydicom
import cv2
import os


def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img


def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # 归一化[0,1]
    img = img * 255  # 0-255
    img = img.astype(np.uint8)
    return img


def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3])
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels


def extend_channels2(img):
    print('Origin')
    print(img)
    img_channels = np.zeros([3, img.shape[0], img.shape[1]])
    img_channels[0, :, :] = img
    img_channels[1, :, :] = img
    img_channels[2, :, :] = img
    print('return')
    print(img_channels, img_channels.shape)
    return img_channels


# 必选：图像预处理组合（基本操作）

def data_preprocess_base(img, size):
    # step1: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step2: 归一化[0,255]
    img = normalize(img)
    # step3: 扩展为3通道 224*224*3
    img = extend_channels(img)
    # Step4: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


def data_preprocess_base2(img, size):
    # step1: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step2: 归一化[0,255]
    img = normalize(img)
    # step3: 扩展为3通道 224*224*3
    img = extend_channels2(img)
    # Step4: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


if __name__ == '__main__':
    dcm_path = "./back/nettest/data/images/00001.dcm"

    absolute_path = os.path.abspath(dcm_path)
    print("绝对路径:", absolute_path)

    dcm_info = pydicom.read_file(dcm_path)
    print(dcm_info)
    dcm_img = dcm_info.pixel_array
    print(dcm_img)
    print('图像形状=', dcm_img.shape,
          '图像最小值=', dcm_img.min(),
          '图像最大值=', dcm_img.max())

    plt.title('dcm_img')
    plt.imshow(dcm_img, cmap='gray')
    plt.show()

    preprocess_base_img = data_preprocess_base(dcm_img, 224)
    plt.title('data_preprocess_base')
    plt.imshow(preprocess_base_img, cmap='gray')
    print('图像形状=', preprocess_base_img.shape,
          '图像最小值=', preprocess_base_img.min(),
          '图像最大值=', preprocess_base_img.max())
    plt.show()

    # 必选：缩放尺寸，默认缩放为224

    # 必选：归一化

    # 必选：单通道扩展为3通道

    extend_channels_img = extend_channels(normalize(dcm_img)).astype(np.uint8)
    plt.title('extend_channels_img')
    plt.imshow(extend_channels_img)
    print(extend_channels_img.shape)
    plt.show()


    def windows_pro(img, min_bound=0, max_bound=85):
        """
            输入：图像，阈值下限min_bound，阈值上限max_bound
            处理过程：先获取指定限制范围内的值[min_bound,max_bound]，再中心化、归一化
            输出：阈值范围缩减后中心化归一化结果[0,255]
        """
        img[img > max_bound] = max_bound
        img[img < min_bound] = min_bound  # [min_bound, max_bound]
        img = img - min_bound  # 中心化[0,max_bound+min_bound]
        img = normalize(img)  # 归一化 [0,255]
        return img


    def equalize_hist(img):
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)
        return img


    # 可选：图像预处理（伪影增强）
    def data_preprocess_enhanced(img, size):
        # step1: 图像阈值范围缩减 [min_bound, max_bound]
        img = windows_pro(img)
        # step2: 直方图均衡 [0, 255]
        img = equalize_hist(img)
        # step3: 缩放尺寸 224*224
        img = img_resize(img, size)
        # step4: 归一化[0,255]
        img = normalize(img)
        # step5: 扩展为3通道 224*224*3
        img = extend_channels(img)
        # Step6: 转换为unit8格式
        img = img.astype(np.uint8)
        return img


    preprocess_enhanced_img = data_preprocess_enhanced(dcm_img, 224)
    plt.title('data_preprocess')
    plt.imshow(preprocess_enhanced_img, cmap='gray')
    print('图像形状=', preprocess_enhanced_img.shape,
          '图像最小值=', preprocess_enhanced_img.min(),
          '图像最大值=', preprocess_enhanced_img.max())
    plt.show()

    windows_pro_img = windows_pro(dcm_img)
    plt.title('windows_pro_img')
    plt.imshow(windows_pro_img, cmap='gray')
    plt.show()

    # 可选：直方图均衡(增加对比度)

    equalize_hist_img = equalize_hist(windows_pro_img)
    # 直方图对比
    # (1)直方图均衡后图像equalize_hist_img的直方图
    plt.hist(equalize_hist_img.ravel(), 20, [equalize_hist_img.min(), equalize_hist_img.max()], color='steelblue')
    plt.title('Histogram of equalize_hist_img')
    plt.tight_layout()
    plt.show()

    # (2)原图dcm_img的直方图
    plt.hist(dcm_img.ravel(), 20, [dcm_img.min(), dcm_img.max()], color='orange')
    plt.title('Histogram of dcm_img')
    # plt.tight_layout()
    plt.show()
