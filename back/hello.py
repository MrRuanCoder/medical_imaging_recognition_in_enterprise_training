'''
Author: Ruan 110537579+MrRuanCoder@users.noreply.github.com
Date: 2023-07-14 20:13:20
LastEditors: Ruan 110537579+MrRuanCoder@users.noreply.github.com
LastEditTime: 2023-07-15 13:51:38
FilePath: \medical_imaging_recognition_in_enterprise_training\back\hello.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from flask import Flask
from flask import request
from flask import abort, redirect
from flask import session
from flask import jsonify
from adminApi import *
from learnSQL import *
from learnSQL import db
import os
from datetime import timedelta

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = '_5#y2L"F4Q8z\n\xec]/1'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///first.db'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7) # 配置7天有效 


# 通过静态路由访问
# 运行程序，http://127.0.0.1:5000/show
# @ app.route("/")
# def show_file():
#     return app.send_static_file("index.html")

@app.route('/')
def mainPage():
    username = session.get('username')
    if username is None:
        return app.send_static_file('index.html')    
    else:
        return app.send_static_file("login.html")


#清空session,退出登录
@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return jsonify(msg='退出成功')    


@app.route('/api', methods=['POST'])
def login():
    try:
        get_data = request.get_json()    
        username = get_data.get('username')
        password = get_data.get('password')
    
        if not all([username,password]):
            return jsonify(code=400, msg='参数不完整')
        
        user = User.query.filter(User.username == username).first() 

        if user is None or password != user.password:
            return jsonify(code=400, msg="账号或密码错误")

        permission = user.permission

            #验证通过，保存登录状态在session中
        session['username'] = username
        session["id"] = user.id
        session["permission"] = permission
            # session['password'] = password
        return jsonify(msg='登录成功')
    
    except Exception as e:
        print(e)
        return jsonify(msg='登录失败')   


# 检查登录状态
@app.route('/session', methods=['GET'])
def check_session():
    username = session.get('username')
    if username is not None:
        #加上操作逻辑，数据库之类的
        return jsonify(username=username)
    else:
        return jsonify(msg='未登录')

# 创建用户 (增)
@app.route("/api/add", methods=["POST"])
def user_add0():
    return user_add()

#获得所有用户（查所有）
@app.route("/api/all", methods=["POST"])
def getAll0():
    return getAll()

#修改用户(改)
@app.route("/api/update", methods=["POST"])
def user_update0():
    return user_update()

#删除用户
@app.route("/api/delete", methods=["DELETE"])
def user_delete0():
    return user_delete()

if __name__ == '__main__':
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.run()
