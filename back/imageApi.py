from flask import jsonify, request, session
from werkzeug.utils import secure_filename
from learnSQL import User, db  
import os
import zipfile

#单个图像处理
def singleImage():
    pass

def zipImage():
    f = request.files['upload']

    savepath = "/opt/upload/"
    # 判断目录是否存在，不存在则新建
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    upload_path = os.path.join(savepath, secure_filename(f.filename))
    f.save(upload_path)

    if zipfile.is_zipfile(upload_path):  # 判断是否zip文件
        zf = zipfile.ZipFile(upload_path, 'r') # 设置文件为可读
        stem, suffix = os.path.splitext(f.filename)  # 提取文件名称
        for file in zf.namelist(): # 遍历文件
            zf.extract(file, savepath + "/" + stem)  # 解压至指定目录