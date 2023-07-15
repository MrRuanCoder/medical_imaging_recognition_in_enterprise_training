| 请求方式 | 说明                           | RequestBody                                                  | url路径                 | 使用flask中的session cookie | ResponseBody                                                 | 备注                                                         |
| -------- | ------------------------------ | ------------------------------------------------------------ | ----------------------- | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| POST     | 验证登录                       | {username:"\u5f20\u533b\u751f",pwd:"aabbcc",remember:0}      | /api/login              | 无                          | 200状态码: {userName:"\u5f20\u533b\u751f",permission:0} 或   200状态码: {userName:"\u5f20\u533b\u751f",permission:1}或  200状态码: {userName:"\u5f20\u533b\u751f",permission:2}或  401状态码 | ResponseBody需要返回utf-8编码后的值,避免编码错乱             |
| GET      | 访问主页                       |                                                              | /                       |                             | 200状态码返回index.html的内容,否则302状态码重定向到login.html |                                                              |
| POST     | 上传含有dcm文件的zip压缩包     | zip压缩包文件                                                | /api/file               |                             | 200状态码: {DiagnosticResults:"\u8111\u762b"}或  401状态码 或  200状态码: {DiagnosticImg:["1689321426/1.png","1689321426/2.png"]} | DiagnosticResults是诊断结果. DiagnosticImg返回一个图片路径的数组,这里的1689321426是时间戳,便于存储图片. |
| POST     | 管理员账户对单个普通账户的增改 | {id:1,pwd:"aabbcc",userName:"\u5f20\u533b\u751f",permission:1} | /api/add或  /api/update |                             | 成功:200状态码;没有权限:401状态码;add时id已重复:400状态码    |                                                              |
| POST     | 管理员账户对单个普通账户的删除 | {id:1}                                                       | /api/delete             |                             |                                                              |                                                              |
| GET      | 管理员账户查询到所有账户       |                                                              | /api/getAll             |                             | 200状态码: {result:[{id:1,userName:"\u5f20\u533b\u751f",pwd:"aabbcc",permission:0},{id:2,userName:"\u5f20\u533b\u751f",pwd:"aabbcc",permission:1}]}或  401状态码 | 对session cookie进行判断,只有管理员账户才能返回200           |
| GET      | 返回登录界面                   |                                                              | /login.html             |                             | 对cookie进行判断,若已登录则302状态码重定向到index.html,否则200状态码返回login.html |                                                              |
| GET      | 返回管理员管理界面             |                                                              | /manage.html            |                             | 对cookie进行判断,若已登录且为管理员账户则200状态码返回manage.html,若已登录但不为管理员账户则302状态码重定向到index.html,若未登录则302状态码重定向到login.html |                                                              |

permission说明:	0是Administrator, 1是Doctor, 2是病人

remember说明:	1是记住密码, 0是不记住密码

# html文档目录

- index.html : 主页,上传文件,返回查询结果

- login.html : 登录页

- manage.html : 管理员账户对账户的增删改查

    

------



# 一个demo

```python

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from PIL import Image


class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        file = open('./ImgPOST.html', 'r', encoding='utf-8').read()
        self.wfile.write(file.encode())

    def do_POST(self):
        print(self.path)
        # 保存接收到的图片文件
        # image_path = 'received_image.jpg'  # 图片保存路径
        # with open(image_path, 'wb') as f:
        #     f.write(post_data)
        #
        # # 获取图片的大小和宽高
        # image = Image.open(image_path)
        # image_size = os.path.getsize(image_path)
        # width, height = image.size
        #
        # # 构建响应消息
        # response_message = f"Image size: {image_size} bytes\n"
        # response_message += f"Image width: {width}px\n"
        # response_message += f"Image height: {height}px"

        # 发送响应消息

        content_type = self.headers['Content-Type']
        if content_type.startswith('multipart/form-data'):
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            content = body.split(b'\r\n\r\n')[1].split(b'\r\n------WebKitFormBoundary')[0]
            # print(content)
            image_path = 'received_image.jpg'  # 图片保存路径
            with open(image_path, 'wb') as f:
                f.write(content)

            # 获取图片的大小和宽高
            image = Image.open(image_path)
            image_size = os.path.getsize(image_path)
            width, height = image.size

            # 构建响应消息
            response_message = f"Image size: {image_size} bytes\n"
            response_message += f"Image width: {width}px\n"
            response_message += f"Image height: {height}px"
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(response_message.encode())

        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Invalid request'.encode())


def run():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, ImageHandler)
    print('Starting server...')
    httpd.serve_forever()


run()

```

C:就是创建(Create), R:就是查找(Retrieve), U:就是更改(Update), D:就是删除(Delete).
