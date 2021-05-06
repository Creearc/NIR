import cv2
import imutils
import numpy as np

def auto_canny(image, sigma=0.33):
  v = np.median(image)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  #edged = cv2.Canny(image, lower, upper)
  edged = cv2.Canny(image, 0, 250)
  return edged

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)


fr = []
out = None

codec = cv2.VideoWriter_fourcc('M','J','P','G')

if __name__ == '__main__':
  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  cap.set(cv2.CAP_PROP_FOURCC, codec)
  cap2 = cv2.VideoCapture(2)
  cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  cap2.set(cv2.CAP_PROP_FOURCC, codec)
  ret = True
  count = 70
  while ret:      
    ret, frame1 = cap.read()
    ret, frame2 = cap2.read()
    if not ret:
      break
    frame = np.hstack([frame1, frame2])
    origin = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = adjust_gamma(gray, 0.3)
    ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    if out is None:
      out = frame.copy()
    if count > 0:
      out = cv2.addWeighted(out, 0.8, frame, 0.2, 0.5)
      count -= 1
      print(count)
    frame = cv2.bitwise_and(cv2.bitwise_not(out), frame)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    ret, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)

    #frame = auto_canny(gray, 1.0)
    if count == 0:
      cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      for c in cnts:
        if cv2.contourArea(c) > 25:
          M = cv2.moments(c)
          if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(origin, (cX, cY), 7, (0, 0, 0), -1)
            cv2.circle(origin, (cX, cY), 5, (255, 255, 255), -1)
          #cv2.drawContours(origin, [c], -1, (0, 255, 255), 2)
        

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = np.vstack([frame, origin])
    cv2.imshow('0', frame)

    #print(cv2.HuMoments(cv2.moments(frame)).flatten())
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
      break
  cv2.destroyAllWindows()
