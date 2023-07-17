from flask import jsonify, request, session
from learnSQL import User, db   #导入数据库模型类和数据库对象
from SQLiteDemo import SQLiteGetAll, modify, add, delete


# 创建用户 (增)
def user_add():
    req_data = request.get_json()
    username = req_data.get("username")
    password = req_data.get("password")
    permission = req_data.get("permission")
    action = req_data.get("action")

    # user = User(username=username, password=password, permission=permission, action=action)
    try:
        if 'permission' in session and session['permission'] != '0':
            return jsonify(code=400, msg="无权限进行操作")
        
        # db.session.add(user)
        # db.session.commit()
        if add(username, password):
            return jsonify(code=200, msg="新增用户成功")
        else:
            return jsonify(code=400, msg="新增用户失败, 可能用户名已存在")
    except Exception as e:
        print(e)
        db.session.rollback()
        return jsonify(code=400, msg="新增补失败")

#获得所有用户（查所有）
def getAll():
    try:
        if session['permission'] != '0':
            return  jsonify(code=400, msg="无权限进行操作")

        # users = User.query.all()  # 查询数据库中所有用户
        users = SQLiteGetAll()
        user_list = []
        for user in users:
            user_data = {
                'id': user[0],
                'username': user[1],
                'password': user[2],
                'permission': user[3],
                'action': user[4]
            }
            user_list.append(user_data)

        return jsonify(users=user_list)
    except Exception as e:
        print(e)
        return jsonify(code=400, msg="获取用户列表失败")
    
#修改用户(改)
def user_update():
    req_data = request.get_json()
    id = req_data.get("id")
    username = req_data.get("username")
    password = req_data.get("password")
    permission = req_data.get("permission")
    action = req_data.get("action")

    try:
        # user = User.query.get(id)  # 查询要更新的用户对象
        # if user is None:
        #     return jsonify(code=400, msg="用户不存在")

        if 'permission' in session and session['permission'] != '0':
            return jsonify(code=400, msg="无权限进行操作")

        # user.username = username  # 更新用户信息
        # user.password = password
        # user.permission = permission
        # user.action = action
        #
        # db.session.commit()  # 提交数据库事务
        if modify(id, username, password) == False:
            return jsonify(code=400, msg="用户信息更新失败, 可能用户名已存在")
        return jsonify(code=200, msg="用户信息更新成功")
    
    except Exception as e:
        print(e)
        db.session.rollback()  # 回滚数据库事务
        return jsonify(code=400, msg="用户信息更新失败")
    
#删除用户
def user_delete():
    req_data = request.get_json()
    user_id = req_data.get("id")  # 获取要删除的用户 ID
    username = req_data.get("username")
    try:
        # user = User.query.get(user_id)  # 查询要删除的用户对象


        if 'permission' in session and session['permission'] != '0':
            return jsonify(code=400, msg="无权限进行操作")
        if delete(user_id)==False:
            return jsonify(code=400, msg="用户无法删除")
        # db.session.delete(user)  # 删除用户对象
        # db.session.commit()  # 提交数据库事务

        return jsonify(code=200, msg="用户删除成功")
    except Exception as e:
        print(e)
        db.session.rollback()  # 回滚数据库事务，保持事务开始前的状态，从而保持数据库的一致性
        return jsonify(code=400, msg="用户删除失败")