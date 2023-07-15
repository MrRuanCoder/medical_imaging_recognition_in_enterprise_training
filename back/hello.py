from flask import Flask
from flask import request
from flask import abort, redirect
from flask import session
from flask import jsonify
from adminApi import *
from learnSQL import *
from learnSQL import db

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = '_5#y2L"F4Q8z\n\xec]/1'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///first.db'


@app.route("/index/<name>")
def hello(name):
    return "Hello World! %s" % (name + name)
 
@app.route('/')
def index():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login.html')
    return 'hello'

@app.route('/')
def mainPage():
    if 'username' in session:
        # 存在会话，返回重定向的目标 URL
        return redirect('/index.html')
    else:
        # 无会话，返回空的重定向 URL
        return redirect('/login.html')


#清空session,退出登录
@app.route('/logout', methods=['GET'])
def logout():
    session.claer()
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
