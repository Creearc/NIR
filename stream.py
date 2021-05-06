import pyrealsense2.pyrealsense2 as rs
from multiprocessing import Process, Queue, Value
import numpy as np
import cv2
import time
import imutils
import os

import requests

import tensorflow as tf
import posenet

cascade = cv2.CascadeClassifier("h0.xml")

HOST = '192.168.68.101'
PORT = 56088

url = 'http://{}:{}'.format(HOST, PORT)

def ask(url, keypoints):
  url = '{}/keypoints'.format(url)
  data = {'keypoints' : keypoints}
  
  response = requests.post(url=url, json=data, timeout=10)
  out = response.json()['status']
  
  return out


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

def process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale

def HAAR(image):
  global cascade
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = cascade.detectMultiScale(image, scaleFactor=1.1,
                                     minNeighbors=2, minSize=(20, 40),
                                     flags = cv2.CASCADE_SCALE_IMAGE)
                     
  if len(rects) == 0: return []
  rects[:,2:] += rects[:,:2]
  box = []
  for x1, y1, x2, y2 in rects:
     box.append((x1, y1, x2, y2))
  return box

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def HOG(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())
  found, w = hog.detectMultiScale(image, winStride=(8,8), padding=(16,16), scale=1.1)
  found_filtered = []
  for ri, r in enumerate(found):
        for qi, q in enumerate(found):
          if ri != qi and inside(r, q): break
          else: found_filtered.append(r)
  box = []
  for x1, y1, w, h in found:
    box.append((x1, y1, x1 + w, y1 + h))
  return box

def get_contours(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
  cnts, hierarchy = cv2.findContours(img,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
  boxes = []
  for cnt in cnts:
    if cv2.contourArea(cnt) < 2000:
      continue
    rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    x1 = min(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    x2 = max(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    y1 = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    y2 = max(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    boxes.append((x1, y1, x2, y2)) 
  return boxes

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results

def get_adjacent_keypoints2(keypoint_scores, keypoint_coords, depth_image, depth_scale, min_confidence=0.1):
    results = dict()
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        coords_1, coords_2 = (np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32))
        try:
          results['{} {}'.format(left, right)] = ([int(coords_1[0]),
                                                   int(coords_1[1]),
                                                   int(depth_image[coords_1[1], coords_1[0]] * depth_scale * 100) / 100],
                                                  
                                                  [int(coords_2[0]),
                                                   int(coords_2[1]),
                                                   int(depth_image[coords_2[1], coords_2[0]] * depth_scale * 100) / 100])
        except:
          continue
    return results

def get_coords(
        img, depth_image, depth_scale, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []

    new_keypoints = {}
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints2(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], depth_image, depth_scale, min_part_score)
        adjacent_keypoints.extend(new_keypoints)
    return new_keypoints

def capture():
  global img_q, save
  w, h = 640, 480
  #w, h = 1280, 720

  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
  
  # Start streaming
  profile = pipeline.start(config)

  # Getting the depth sensor's depth scale (see rs-align example for explanation)
  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: " , depth_scale)


  clipping_distance_in_meters = 2 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale


  frame_count = 0
  t = time.time()

  with tf.Session() as sess:
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']
    while True:
      
      frames = pipeline.wait_for_frames()
      """
      aligned_frames = align.process(frames)

      aligned_depth_frame = aligned_frames.get_depth_frame()
      color_frame = aligned_frames.get_color_frame()
          
      if not aligned_depth_frame or not color_frame:
        continue

      depth_image = np.asanyarray(aligned_depth_frame.get_data())
      color_image = np.asanyarray(color_frame.get_data())"""

      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()

      if not depth_frame or not color_frame:
        continue

      depth_image = np.asanyarray(depth_frame.get_data())#[:, 300 : 900]
      color_image = np.asanyarray(color_frame.get_data())#[:, 300 : 900]
      
      
      out = np.zeros((h, w, 3),np.uint8)
      
      depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
      out = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 0, color_image)
      
      mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0, 255)
      mask = cv2.applyColorMap(cv2.convertScaleAbs(mask, alpha=1), cv2.COLORMAP_BONE)
      blur_p = 5
      mask = cv2.GaussianBlur(mask,(blur_p, blur_p), 0)
      
      #mask = np.dstack((mask,mask,mask))
      boxes = []
      boxes = get_contours(mask.copy())
      #boxes = []
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)
      #boxes = HAAR(color_image)
      keypoints = dict()
      for box in boxes:
          img = out[box[1] : box[3], box[0] : box[2]]
          img2 = color_image[box[1] : box[3], box[0] : box[2]]
          if img.shape[0] == 0 or img.shape[1] == 0:
            continue
          input_image, display_image, output_scale = process_input(img, scale_factor=0.7125, output_stride=output_stride)
          
          heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image})
  
          pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)

          keypoint_coords *= output_scale

          tmp = get_coords(
                        img2, depth_image, depth_scale, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0.15, min_part_score=0.1)
          if len(tmp.keys()) > len(keypoints.keys()):
            keypoints = tmp.copy()
          
        
      
      if len(boxes) > 0:
        ask(url, keypoints)
      
      

p = Process(target=capture, args=())
p.start()
