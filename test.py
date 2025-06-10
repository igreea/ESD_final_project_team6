import socket

HOST="192.168.10.1"
PORT=5000


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST,PORT))

client_socket.sendall(b'Hello, labtop!')
data = client_socket.recv(1024)

print(f"[수신] {data.decode()}")
client_socket.close()
