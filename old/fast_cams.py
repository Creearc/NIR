import cv2
import imutils
import numpy as np

import threading

codec = cv2.VideoWriter_fourcc('M','J','P','G')
W, H = 1280, 720
#W, H = 640, 480

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

lock = threading.Lock()

class WebcamVideoStream:
    def __init__(self, src='/dev/video0'):
        print('Camera %s init...' % src)
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)           
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

def record():
    global c

    count = 50
    mask = None
    
    out = np.zeros((H * 2, W * len(c), 3), np.uint8)
    
    video = []
    video.append(cv2.VideoWriter('0.avi', codec, 60.0, (W, H)))
    video.append(cv2.VideoWriter('1.avi', codec, 60.0, (W, H)))
    while True:
        for i in range(len(c)):
            with lock:
                img = c[i].read()
            if img is None:
                continue
            video[i].write(img)
            gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
            frame = adjust_gamma(gray, 0.4)
            ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

            if mask is None:
              mask = frame.copy()
            if count > 0:
              mask = cv2.addWeighted(mask, 0.9, frame, 0.1, 0.8)
              count -= 1
              print(count)
            frame = cv2.bitwise_and(cv2.bitwise_not(mask), frame)
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
            ret, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)

            if count == 0:
              cv2.circle(img, (10, 10), 15, (0, 255, 0), -1)
              cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
              cnts = imutils.grab_contours(cnts)
              for cnt in cnts:
                if cv2.contourArea(cnt) > 5:
                  M = cv2.moments(cnt)
                  if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(img, (cX, cY), 15, (0, 0, 255), -1)
                    cv2.circle(img, (cX, cY), 10, (255, 0, 0), -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            out[0 : H, W * i : W * (i + 1)] = img
            out[H : 2 * H, W * i : W * (i + 1)] = frame
            
        cv2.imshow('0', imutils.resize(out, width=1920, inter=cv2.INTER_NEAREST))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break


c = []

if __name__ == '__main__':
    c.append(WebcamVideoStream('7 (1).avi').start())
    c.append(WebcamVideoStream('7 (2).avi').start())
    tr = threading.Thread(target=record, args=())
    tr.start()
