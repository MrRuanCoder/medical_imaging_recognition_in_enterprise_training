######################################################DISCARD################################################
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@127.0.0.1:3306/test'

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + "/home/lmp/test.db"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'xxx'
db = SQLAlchemy(app)

# 用户表
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)  # id号(独一无二的)
    username = db.Column(db.String(64), nullable=False, autoincrement=True)  # 姓名
    password = db.Column(db.String(255), nullable=False)  
    permission = db.Column(db.Enum(0, 1), nullable=False)  
    action = db.Column(db.String(64), nullable=False)  

if __name__ == "__main__":
    db.create_all()
    # db.drop_all()