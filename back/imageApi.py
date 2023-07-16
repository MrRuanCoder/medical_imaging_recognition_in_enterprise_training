
from flask import jsonify, request, session
from werkzeug.utils import secure_filename
# from learnSQL import User, db  
import os
import zipfile
from werkzeug.utils import secure_filename
from flask import render_template
import socket
import pydicom
from io import BytesIO
import numpy
from PIL import Image

from nettest import predict_test

#单个图像处理
def singleImage():
    f = request.files['file']
    savepath = "./opt/upload/"
    # 判断目录是否存在，不存在则新建
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

def zipImage1():
    f = request.files['file']

    savepath = "./opt/upload/"
    # 判断目录是否存在，不存在则新建
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    upload_path = os.path.join(savepath, secure_filename(f.filename))
    f.save(upload_path)

    if zipfile.is_zipfile(upload_path):  # 判断是否为zip文件
        zf = zipfile.ZipFile(upload_path, 'r')  # 设置文件为可读
        stem, suffix = os.path.splitext(f.filename)  # 提取文件名称
        target_dir = os.path.join(savepath, stem)  # 指定目录
        os.makedirs(target_dir, exist_ok=True)  # 创建目标目录
        zf.extractall(target_dir)  # 解压至指定目录
        zf.close()

    return jsonify(msg='文件上传成功')

def zipImage2(): 
    pass

def transformImage():
    image_path = '../opt/upload/00001.dcm'  # DICOM文件路径
    dicom_data = pydicom.dcmread(image_path)
    image_data = dicom_data.pixel_array.tobytes()

# IPV4/TCP 协议
    tcp_client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print(tcp_client_socket)
# 2 和服务器端建立连接
    # tcp_client_socket.connect(('10.203.98.3',8080))
    tcp_client_socket.connect(('127.0.0.1',8080))
# 3 发送数据，必须是字节流
# encode(把字符串转为bytes) decode（把bytes转为字符串）
#     # while True:
#     data = input('发送')
#         # if data != '*':
#     data = data.encode('gbk')
#     tcp_client_socket.send(data)
#     # 4 接收服务器返回的数据  recv(字节大小）
#     recv_data = tcp_client_socket.recv(1024).decode('gbk')
#     print('服务器：', recv_data)
# # 5 关闭套接字对象
#         # else:
#     tcp_client_socket.close()
#     exit()

# 发送数据
    tcp_client_socket.send(image_data)

# # 接收服务器返回的数据
#     recv_data = tcp_client_socket.recv(1024*1024*10).decode('gbk')
#     # print('服务器：', recv_data)
#     print(tcp_client_socket)
#     # recv_data = tcp_client_socket.recv(1024*1024*100)
#     print( recv_data)
#     try:
#         recv_data = recv_data.decode('gbk')
#         print('服务器：', recv_data)
#     except UnicodeDecodeError:
#         # 如果解码出错，尝试使用其他编码方式进行解码
#         recv_data = recv_data.decode('utf-8')
#         print('服务器（使用UTF-8解码）：', recv_data)

#     try:
#         dcm_data = pydicom.dcmread(BytesIO(recv_data))
#         if dcm_data.file_meta.FileMetaInformationVersion:
#             print('服务器返回的数据是DICOM图片')
#         # 在这里添加处理DICOM图片的逻辑
#         else:
#             print('服务器返回的数据不是DICOM图片')
#         # 在这里添加处理非DICOM图片的逻辑
#     except pydicom.errors.InvalidDicomError:
#         print('服务器返回的数据不是DICOM图片')
#     # 在这里添加处理非DICOM图片的逻辑

# # 关闭套接字对象
#     tcp_client_socket.close()

    # 接收服务端返回的PNG图像数据
    png_data = b""
    while True:
        recv_data = tcp_client_socket.recv(1024*1024)
        if not recv_data:
            break
        png_data += recv_data

    # 保存PNG图像数据为本地文件
    output_path = './received_image.png'
    with open(output_path, 'wb') as file:
        file.write(png_data)

    # 关闭套接字对象
    tcp_client_socket.close()

    # 显示接收到的PNG图像
    image = Image.open(output_path)
    image.show()


if __name__ == '__main__':
    transformImage()
