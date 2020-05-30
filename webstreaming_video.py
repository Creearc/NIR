import os
import sys
import threading
import argparse
import time

import numpy as np
import cv2
import imutils

import paho.mqtt.client as paho
from paho.mqtt import publish


codec = cv2.VideoWriter_fourcc('M','J','P','G')
#codec = cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V')

class WebcamVideoStream:
    def __init__(self, src='/dev/video0'):
        print('Camera %s init...' % src)
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args["width"])
        self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args["height"])
        self.cap.set(cv2.CAP_PROP_FPS, args["fps"])           
        (self.grabbed, self.frame) = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def get(self, var1):
        self.cap.get(var1)
        
    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            #s = time.time()
            self.grabbed, self.frame = self.cap.read()
            #print(1/(time.time() - s))

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


def send():
    global c, lock
    #client1 = paho.Client(client_id="pub-" + IP)
    #client1.connect(BROKER, PORT)
    h, w = args["height"] // args["frame_size"], args["width"] // args["frame_size"]
    if args["is_black"] == 1:
        out = np.zeros((h, w * len(c)), np.uint8)
    else:
        out = np.zeros((h, w * len(c), 3), np.uint8)
    img = None
    print('Streaming started...')
    while True: 
        for i in range(len(c)):
            with lock:
                img = c[i].read()
            if img is None:
                continue
            img = imutils.resize(img, width=w, inter=cv2.INTER_NEAREST)
            if args["is_black"] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out[0 : h, w * i : w * (i + 1)] = img
        (flag, encodedImage) = cv2.imencode(".jpg", out)
        if not flag:
            continue
        publish.multiple([{'topic': IP, 'payload': bytearray(encodedImage)}], hostname=BROKER, port=PORT)
        #client1.publish(IP, bytearray(encodedImage), qos=0, retain=False)

def record():
    global c
    while True:
        l = len(os.listdir('Video/'))
        h, w = args["height"], args["width"]
        video = []
        v = []
        f = []
        for i in range(len(c)):
            v.append(l + i)
            video.append(cv2.VideoWriter('Video/' + str(v[i]) +'.avi', codec, args["quality"], (w, h)))
            print('Recording to ' + 'Video/' + str(v[i]) +'.avi')
            f.append(0)
        while True:
            for i in range(len(c)):
                with lock:
                    img = c[i].read()
                if img is None:
                    continue
                f[i] += 1
                if f[i] % 1000 == 0:
                    size = os.popen("du Video/" + str(v[i]) + '.avi').read().split('\t')[0]
                    print('{}: {}b'.format(i, size))
                    if int(size) > 1900000:
                        break
                video[i].write(img)

		
outputFrame = None
lock = threading.Lock()
out = None
c = []

BROKER = '192.168.9.8'
PORT = 56008

IP = os.popen("""ifconfig eth0 | grep inet | awk '{ print $2 }'""").read().split('\n')[0]
print('My IP is %s' % IP)

def found_cameras(c):
    print("Searching for cameras...")
    out = os.popen("v4l2-ctl --list-devices").read().split('\n')
    for i in range(len(out)):
        if out[i].find('CAMERA') != -1:
            camera = out[i + 1].strip()
            print('Camera %s founded' % camera)
            c.append(WebcamVideoStream(camera).start())
    print('All cameras connected.')

        

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, default='0.0.0.0',
		help="ip address of the device")
	ap.add_argument("-p", "--port", type=int, default=8000,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-s", "--frame_size", type=int, default=1)
	ap.add_argument("-b", "--is_black", type=int, default=1)
	ap.add_argument("-q", "--quality", type=float, default=90.0)
	ap.add_argument("-w", "--width", type=int, default=1280)
	ap.add_argument("-u", "--height", type=int, default=720)
	ap.add_argument("-f", "--fps", type=int, default=20)
	ap.add_argument("-c", "--cameras", type=int, default=1)
	ap.add_argument("-v", "--record", type=int, default=0)
	args = vars(ap.parse_args())

	found_cameras(c)
	
	if args["record"] > 0:
            tr = threading.Thread(target=record, args=())
            tr.start()
            
	tr1 = threading.Thread(target=send, args=())
	tr1.start()
        
