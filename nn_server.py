from flask import Flask, request, redirect, url_for, jsonify
import json
import requests
import os

import time

def get_coords(d, p):
  for key in d.keys():
    s = key.split()
    for i in range(len(s)):
      p[int(s[i])] = d[key][i]
  return p

app = Flask(__name__)

@app.route('/keypoints', methods=['POST'])
def get_keypoints():
  global t, pos
  print(1 / (time.time() - t))
  if request.method == 'POST':
    data = request.data
    info = json.loads(data)['keypoints']
    print(info)
    if info != {}:
      pos = get_coords(info, pos)
    t = time.time()
    return jsonify({'status' : 1})

@app.route('/pos', methods=['POST'])
def send_pos():
  global t, pos
  print(1 / (time.time() - t))
  if request.method == 'POST':
    print(pos)
    t = time.time()
    return jsonify({'pos' : pos})

t = time.time()

pos = dict()
for i in range(17):
  pos[i] = [0, 0, 0]

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=56088, debug=True)
