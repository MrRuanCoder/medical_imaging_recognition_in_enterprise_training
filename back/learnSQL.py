'''
Author: Ruan 110537579+MrRuanCoder@users.noreply.github.com
Date: 2023-07-15 08:12:29
LastEditors: Ruan 110537579+MrRuanCoder@users.noreply.github.com
LastEditTime: 2023-07-15 10:18:58
FilePath: \project\app\learnSQL.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Ruan 110537579+MrRuanCoder@users.noreply.github.com
Date: 2023-07-15 08:12:29
LastEditors: Ruan 110537579+MrRuanCoder@users.noreply.github.com
LastEditTime: 2023-07-15 09:19:00
FilePath: \project\app\learnSQL.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# pip install flask-sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import pymysql

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@127.0.0.1:3306/test'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + "./first.db"  #具体的位置
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True    #True。当对象被修改时，SQLAlchemy 会发出信号  False 可以关闭这种追踪行为，从而提高应用程序的性能。
app.config['SECRET_KEY'] = 'xxx'        #密钥
db = SQLAlchemy(app)    

# 用户表
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)  # id号(独一无二的)
    username = db.Column(db.String(64), nullable=False, autoincrement=True)  # 姓名
    password = db.Column(db.String(255), nullable=False)  
    permission = db.Column(db.Enum('0', '1'), nullable=False)  
    action = db.Column(db.String(64), nullable=False)  
    # gender = db.Column(db.Enum("男", "女"), nullable=False)  # 学生性别

if __name__ == "__main__":
    db.create_all()