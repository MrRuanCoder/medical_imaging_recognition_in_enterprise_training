from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from PIL import Image


class ImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        file = open('./index.html', 'r', encoding='utf-8').read()
        self.wfile.write(file.encode())

    def do_POST(self):
        print(self.path)
        if not self.path=='/api/file':
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Invalid request'.encode())
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
            image_path = 'received_image.zip'  # 图片保存路径
            with open(image_path, 'wb') as f:
                f.write(content)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('yes'.encode())
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
            # self.send_response(200)
            # self.send_header('Content-type', 'text/plain')
            # self.end_headers()
            # self.wfile.write(response_message.encode())

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
