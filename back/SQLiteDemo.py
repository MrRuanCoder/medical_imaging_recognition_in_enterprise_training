import sqlite3

def query(username):
    try:
        c = sqlite3.connect('first.db')
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
    try:
        c = sqlite3.connect('first.db')
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
    try:
        c = sqlite3.connect('first.db')
        cursor = c.execute(f"UPDATE user SET username='{username}',password='{password}',permission='{permission}' WHERE id='{id}'")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def add(username,password,permission):
    try:
        c = sqlite3.connect('first.db')
        cursor = c.execute(f"INSERT INTO user (username,password,permission,action) VALUES ('{username}','{password}','{permission}','1')")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()
def delete(id):
    try:
        c = sqlite3.connect('first.db')
        cursor = c.execute(f"DELETE FROM user WHERE id='{id}'")
        c.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        c.close()