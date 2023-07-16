import sqlite3
import os
def realpath():
    path1 = os.path.dirname(__file__) + os.sep + 'first.db'
    path2 = os.getcwd() + os.sep + 'first.db'
    path = ''
    print('path1: ', path1)
    print('path2: ', path2)
    if (os.path.exists(path1)):
        path = path1
    elif (os.path.exists(path2)):
        path = path2
    return path
def query(username):
    global c
    try:
        c = sqlite3.connect(realpath())
        cursor = c.execute(f"SELECT * FROM user WHERE username='{username}'")
        c.commit()
        arr=[]
        for row in cursor:
            # print(row)
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
def modify(id,username,password,permission):
    global c
    try:
        c = sqlite3.connect(realpath())
        c.execute(
            f"UPDATE user SET username='{username}',password='{password}',permission='{permission}' WHERE id='{id}'")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def add(username,password,permission):
    global c
    try:
        c = sqlite3.connect(realpath())
        c.execute(
            f"INSERT INTO user (username,password,permission,action) VALUES ('{username}','{password}','{permission}','1')")
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
        cursor = c.execute(f"DELETE FROM user WHERE id='{id}'")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()