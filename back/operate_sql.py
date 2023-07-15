'''
Author: Ruan 110537579+MrRuanCoder@users.noreply.github.com
Date: 2023-07-15 08:46:57
LastEditors: Ruan 110537579+MrRuanCoder@users.noreply.github.com
LastEditTime: 2023-07-15 09:17:12
FilePath: \project\app\operate_sql.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from learnSQL import db, User

#增
def add():
    s = User( username="Alex", password="12345678900", permission='0', action='1')
    db.session.add(s)
    db.session.commit()
# db.session.add_all([s1, s2, s3, s4])

#查
# get(id) 查单一个
# stu = User.query.get(1)
# print(stu.username)
# all() 查全部
# stu = Student.query.all()
# for i in stu:
#     print(i.name, i.gender, i.phone)
# filter() 条件查询
# stu = Student.query.filter(Student.gender == "女")
# for i in stu:
#     print(i.name, i.id,i.gender)


#改
# 第一种
# stu = Student.query.filter(Student.id == 1).update({"name": "张毅"})# 返回动了多少条数据
# db.session.commit()

# 第二种
# stu = Student.query.filter(Student.gender == "女").all()
# for i in stu:
#     i.gender = "男"
#     db.session.add(i)
# db.session.commit()

#删
# stu = Student.query.filter(Student.id >= 5).delete()  # 返回动了多少条数据
# print(stu)
# db.session.commit()

if __name__ == '__main__':
    add()