from flask import Flask, request, redirect, url_for, jsonify
import json
import requests
import os

import time
import math

pixel_to_m = lambda x, d, w: (x * d) / math.sqrt(x ** 2 + (w / 2) ** 2)
pixel_to_m = lambda x, d, w: x / 20

def get_coords(d, p):
  for key in d.keys():
    p[int(key)] = d[key]
  return p

def recompute(p):
  for key in p.keys():
    x, y, z = p[key]
    x = pixel_to_m(x - 320, z, 640)
    y = pixel_to_m(y - 240, z, 480)
    p[key] = [x, y, z]
  return p

app = Flask(__name__)

@app.route('/keypoints', methods=['POST'])
def get_keypoints():
  global t, pos
  print(1 / (time.time() - t))
  if request.method == 'POST':
    data = request.data
    info = json.loads(data)['keypoints']
    print('Received\n{}'.format(info))
    if info != {}:
      pos = get_coords(info, pos)
      pos = recompute(pos)
    t = time.time()
    return jsonify({'status' : 1})

@app.route('/pos', methods=['POST'])
def send_pos():
  global t, pos
  print(1 / (time.time() - t))
  if request.method == 'POST':
    print('Computed\n{}'.format(pos))
    t = time.time()
    return jsonify({'pos' : pos})

t = time.time()

pos = dict()
for i in range(17):
  pos[i] = [0, 0, 0]

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=56088, debug=True)
