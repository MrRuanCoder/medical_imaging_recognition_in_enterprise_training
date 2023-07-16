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