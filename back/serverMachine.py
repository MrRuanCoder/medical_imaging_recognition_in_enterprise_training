import time
import uuid
import pydicom
import socket
import threading

def handle_client_request(service_client_socket, ip_port):
    # 接收图片数据
    image_data = b""
    while True:
        recv_data = service_client_socket.recv(1024)
        if not recv_data:
            break
        image_data += recv_data
    
    # 判断接收到的数据类型
    if image_data.startswith(b'DICOM'):  # 如果是DICOM格式的图片数据
        # 生成唯一的文件名
        image_path = f'./received_image_{time.time()}_{uuid.uuid4()}.dcm'
        with open(image_path, 'wb') as file:
            file.write(image_data)
        
        # 读取DICOM文件
        dcm_data = pydicom.dcmread(image_path)

        # 进行相应的处理操作
        # 在这里添加你想要进行的处理操作
        # 例如，可以获取DICOM属性，修改像素数据等

        # 发送保存下来的DICOM图片数据给客户端
        # with open(image_path, 'rb') as file:
        #     dcm_data = file.read()
        #     # 打印发送的数据大小
        #     print("发送的数据大小:", len(dcm_data))
        #     service_client_socket.send(dcm_data)        

        # 读取PNG文件数据
        with open('../opt/upload/output.png', 'rb') as file:
            png_data = file.read()

        # 发送保存下来的PNG图片数据给客户端
        service_client_socket.send(png_data)

        # # 将处理后的DICOM数据转换为字节流
        # processed_image_data = dcm_data.pixel_array.tobytes()

        # # 发送处理后的图片数据给客户端
        # service_client_socket.send(processed_image_data)
        
    else:
        print(image_data.decode('gbk'), ip_port)  
        back = f'自动回复：欢迎，您发送的数据为{image_data.decode("gbk")}'
        service_client_socket.send(back.encode('gbk'))
    
    # 关闭套接字对象
    service_client_socket.close()


if __name__ == '__main__':
    # 创建套接字对象
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口号复用
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    # 绑定IP和端口，ip不写或写'127.0.0.1'都是主机本身
    tcp_server_socket.bind(('', 8080))
    # 设置监听，一般为5，最大为128
    tcp_server_socket.listen(5)

    # while True:
    # 接收客户端套接字对象
    service_client_socket, ip_port = tcp_server_socket.accept()
    # 创建线程，传入套接字对象数据
    sub_thread = threading.Thread(target=handle_client_request, args=(service_client_socket, ip_port))
    # 守护主线程
    sub_thread.setDaemon(True)
    # 运行线程
    sub_thread.start()