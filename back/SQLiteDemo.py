import sqlite3

def query(username):
    try:
        c = sqlite3.connect('first.db')
        cursor = c.execute(f"SELECT * FROM user WHERE username='{username}'")
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

query("Admin")