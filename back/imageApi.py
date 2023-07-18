from flask import jsonify, request, session
from werkzeug.utils import secure_filename
# from learnSQL import User, db  
import os
import time
import zipfile
import shutil
from werkzeug.utils import secure_filename
from flask import render_template
from flask import send_file
import socket
import pydicom
from io import BytesIO
import numpy as np
from PIL import Image

from predict_test import test_alexnet, output_alexnet
from picture_test import image_deal_transfer 


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

    dcm_filenames = []

    if zipfile.is_zipfile(upload_path):  # 判断是否为zip文件
        zf = zipfile.ZipFile(upload_path, 'r')  # 设置文件为可读
        stem, suffix = os.path.splitext(f.filename)  # 提取文件名称
        # target_dir = os.path.join(savepath, stem)  # 指定目录
        # 使用当前时间戳作为文件夹名
        timestamp = str(int(time.time()))
        target_dir = os.path.join(savepath, timestamp)  # 指定目录
        os.makedirs(target_dir, exist_ok=True)  # 创建目标目录
        zf.extractall(target_dir)  # 解压至指定目录
        zf.close()

        # 遍历目标目录，获取所有dcm文件的名称
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dcm_filename = os.path.join(root, file)
                    dcm_filenames.append(dcm_filename)
        
        
    length = len(dcm_filenames)

    predictoutput=[]    #Tensor 类型不可JSON 可序列化

    for(dcm_filename, i) in zip(dcm_filenames, range(length)):
        # predictoutput.append(output_alexnet('model/L1_model.pkl', dcm_filename))
        # print(output_alexnet('model/L1_model.pkl', dcm_filename))
        tensor_output = output_alexnet('model/L3_resnet18_best_model.pkl', dcm_filename)
        predictoutput.append(tensor_output.tolist())
        image_deal_transfer(dcm_filename)

    # return jsonify(msg='文件上传成功')
    return jsonify(msg='文件上传成功', dcm_filenames=dcm_filenames, length=length, predictoutput=predictoutput)

def zipDownload(): 
# 假设前端发送的请求中包含的相对路径文件列表存储在files变量中
    # files = [
    #     './static/file1.txt',
    #     './static/file2.txt',
    #     './static/file3.txt'
    # ]
    
    data = request.get_json()
    files = data.get('fileList', [])  # 获取前端请求中的文件列表，默认为空列表

    if len(files) == 1 and files[0].endswith('.dcm'):
        # 当只有一个DCM文件时，复制到临时目录并返回该文件的相对路径
        file = files[0]
        file_name = file.split('/')[-1]  # 提取文件名
        destination = f'./static/temp/{file_name}'
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(file, destination)
        return destination

    # 创建一个临时目录来保存待打包的文件
    temp_dir = f'./static/temp/{int(time.time())}'  # 使用时间戳创建一个唯一的文件夹路径
    os.makedirs(temp_dir, exist_ok=True)    #exist_ok存在是否会引发异常

    # # 复制文件到临时目录
    for file in files:
        if file.endswith('.dcm'):
            file_name = file.split('/')[-1]  # 提取文件名
            destination = os.path.join(temp_dir, file_name)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(file, destination)

    # 创建以时间戳命名的ZIP文件
    timestamp = str(int(time.time()))
    zip_file_path = f'./static/temp/files_{timestamp}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, temp_dir))

    # # 创建zip文件
    # zip_file_path = './temp/files.zip'
    # with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
    #     for root, _, files in os.walk(temp_dir):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             zip_file.write(file_path, os.path.relpath(file_path, temp_dir))

    # 删除临时目录
    shutil.rmtree(temp_dir)

    # 发送zip文件给前端
    # return send_file(zip_file_path, as_attachment=True, attachment_filename='files.zip')
    #返回ZIP文件的相对路径
    return zip_file_path
    # return os.path.abspath(zip_file_path)
    # relative_path = os.path.relpath(zip_file_path, './')
    # return jsonify({'path': relative_path})

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
