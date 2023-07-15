from flask import jsonify, request, session
from werkzeug.utils import secure_filename
from learnSQL import User, db  
import os
import zipfile
from werkzeug.utils import secure_filename
from flask import render_template

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