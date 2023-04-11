import os
import json
import torch
import tempfile
from datetime import datetime
import socket
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def send_data(data_to_send):
    action_bytes = bytes(str(data_to_send), "ascii")
    client_socket.send(action_bytes)

def receive_images():
    data_string = (get_data().decode('utf-8'))
    print(data_string)
    screenshots = json.loads(data_string)
    screenshots = screenshots["screenshots"]
    imgs = {}
    for item in screenshots:
        img_path = os.path.join(images_path, item["screenshotPath"])
        imgs[item["junctionId"]] = cv2.imread(img_path)
    return imgs

def prepro(x):
    x = cv2.resize(x, (100, 100))
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255
    x = torch.from_numpy(x)
    return (x.unsqueeze(0).type(torch.cuda.FloatTensor))

def get_data():
    return client_socket.recv(1024)

port = 13000
images_path = os.path.join(tempfile.gettempdir(), "Traffic3D_Screenshots", datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f"))
os.makedirs(images_path, exist_ok=True)

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.bind(("0.0.0.0", port))
ss.listen()
print("waiting for tcpConnection")
(client_socket, address) = ss.accept()
print("tcpConnection established")
send_data(images_path)
max_number_of_junction_states = int(get_data().decode('utf-8'))
for i in range(1, 3):
    imgs = receive_images()
    img = imgs["1"]
    # img = prepro(img)
    testimg = cv2.imread("C://Users/win10/Desktop/test.png")
    # testimg = prepro(testimg)    
    img = cv2.subtract(img, testimg)
    img = prepro(img)
    plt.figure()
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.show()
##    for junction_id in imgs:    
##        img1 = prepro(imgs[junction_id])
##        testimg = cv2.imread("C://Users/win10/Desktop/test.png")
##        testimg = prepro(testimg)
##        img = img1 - testimg
##        plt.figure()
##        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
##        plt.title('Sample image')
##        plt.show()

