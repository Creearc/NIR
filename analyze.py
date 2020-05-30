import cv2
import imutils
import numpy as np

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

def move(image, background, bks):
  image = imutils.resize(image, width=bks, inter=cv2.INTER_NEAREST)
  return out

vnum = 2
video = []
w, h = None, None

count = 100
mask = None
static = None

background = None
bks = 32

for i in range(vnum):
  video.append(cv2.VideoCapture('8 ({}).avi'.format(i+1)))

ret = True

while ret:
  for i in range(vnum):
    ret, img = video[i].read()
    if not ret:
      break

    if w is None:
      h, w = img.shape[:2]
      out = np.zeros((h * 2, w * vnum, 3), np.uint8)

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    frame = adjust_gamma(gray, 0.5)
    ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mask is None:
      mask = frame.copy()
      static = img.copy()
    if count > 0:
      mask = cv2.addWeighted(mask, 0.9, frame, 0.1, 0.8)
      static = cv2.addWeighted(static, 0.5, img, 0.5, 0.1)
      count -= 1
      print(count)
    frame = cv2.bitwise_and(cv2.bitwise_not(mask), frame)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    ret, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)

    img = cv2.bitwise_and(cv2.bitwise_not(static), img)
    img= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #frame = cv2.GaussianBlur(frame, (7, 7), 0)
    
    if count == 0:
      cv2.circle(img, (10, 10), 15, (0, 255, 0), -1)
      cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      for cnt in cnts:
        if cv2.contourArea(cnt) > 25:
          M = cv2.moments(cnt)
          if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 15, (0, 0, 255), -1)
            cv2.circle(img, (cX, cY), 10, (255, 0, 0), -1)
            
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    out[0 : h, w * i : w * (i + 1)] = img
    out[h : 2 * h, w * i : w * (i + 1)] = frame

  cv2.imshow('0', imutils.resize(out, width=1920, inter=cv2.INTER_NEAREST))
  key = cv2.waitKey(0) & 0xFF
  if key == ord("q") or key == 27:
    break

cv2.destroyAllWindows()
