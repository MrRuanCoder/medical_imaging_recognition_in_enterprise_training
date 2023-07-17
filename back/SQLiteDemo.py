import sqlite3
import os
def realpath():
    path1 = os.path.dirname(__file__) + os.sep + 'first.db'
    path2 = os.getcwd() + os.sep + 'first.db'
    path = ''
    # print('path1: ', path1)
    # print('path2: ', path2)
    if (os.path.exists(path1)):
        path = path1
    elif (os.path.exists(path2)):
        path = path2
    return path
def query(username):
    global c
    try:
        c = sqlite3.connect(realpath())
        cursor = c.execute("SELECT * FROM user WHERE username=?", (username,)) #防止sql注入
        arr = []
        for row in cursor:
            arr.append(row)
        return arr
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()

def SQLiteGetAll():
    global c
    try:
        c = sqlite3.connect(realpath())
        cursor = c.execute("SELECT * FROM user")
        c.commit()
        arr=[]
        for row in cursor:
            arr.append(row)
        return arr
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def modify(id, username, password):
    global c
    try:
        c = sqlite3.connect(realpath())
        cursor = c.cursor()

        # 查询数据库中除了当前 ID 对应的记录外，是否已存在相同的用户名
        query = f"SELECT username FROM user WHERE username = '{username}' AND id != '{id}'"
        cursor.execute(query)
        existing_username = cursor.fetchone()

        if existing_username:
            # 如果存在相同的用户名，返回 False
            return False
        c.execute(
            f"UPDATE user SET username='{username}',password='{password}' WHERE id='{id}'")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def add(username, password):
    global c
    try:
        c = sqlite3.connect(realpath())
        cursor = c.cursor()

        # 查询数据库中是否已存在相同的用户名
        querySQL = f"SELECT username FROM user WHERE username = '{username}'"
        cursor.execute(querySQL)
        existing_username = cursor.fetchone()

        if existing_username:
            # 如果相同的用户名已经存在，返回 False
            return False
        c.execute(
            f"INSERT INTO user (username,password,permission,action) VALUES ('{username}','{password}','1','1')")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def delete(id):
    global c
    try:
        c = sqlite3.connect(realpath())
        c.execute(f"DELETE FROM user WHERE id='{id}' AND permission!='0'")
        c.commit()
        rows_deleted = c.total_changes
        print(rows_deleted)
        return rows_deleted > 0
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()