import cv2
import imutils
import numpy as np


def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

video = cv2.VideoCapture('8 (1).avi')
ret = True

count = 90

w, h = None, None
background = None
bks = 640

while ret:
  ret, img = video.read()
  if not ret:
    break

  if w is None:
    h, w = img.shape[:2]

  frame = img.copy()
  frame = imutils.resize(frame, width=bks, inter=cv2.INTER_NEAREST)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = cv2.GaussianBlur(frame, (3, 3), 0)

  if background is None:
    background = frame.copy()
    continue
  if count > 0:
    background = cv2.addWeighted(background, 0.9, frame, 0.1, 0.8)
    count -= 1
    print(count)

  mask = (cv2.absdiff(frame, background))
  
  mask2 = mask.copy()
  _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
  mask = cv2.GaussianBlur(mask, (3, 3), 0)
  _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

  debug = imutils.resize(np.hstack([frame, background, mask2, mask]), width=w, inter=cv2.INTER_NEAREST)
  debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
  
  mask = imutils.resize(mask, width=w, inter=cv2.INTER_NEAREST)
  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

  img = cv2.bitwise_and(mask, img)

   
  
  cv2.imshow('0', imutils.resize(np.vstack([img, debug]), height=1000, inter=cv2.INTER_NEAREST))
  key = cv2.waitKey(0) & 0xFF
  if key == ord("q") or key == 27:
    break
cv2.destroyAllWindows()
