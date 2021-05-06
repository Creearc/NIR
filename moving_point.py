from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import random

import requests

def ask(url):
  url = '{}/pos'.format(url)

  response = requests.post(url=url, timeout=10)
  out = response.json()['pos']

  return out

HOST = '192.168.68.101'
PORT = 56088

url = 'http://{}:{}'.format(HOST, PORT)

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

fig = plt.figure()
ax = p3.Axes3D(fig)

cols = 'rgbcmy'
r = 30

l = 17

x = np.array(range(l))
y = np.array(range(l))
z = np.array(range(l))
c = np.array(range(l))

points, = ax.plot(x, y, z, 'k')

#texts, = ax.text(x, y, z, PART_NAMES, color='red')


max_x = 10
max_y = 10
max_z = 10

ax.set_xlim([0, max_x])
ax.set_ylim([0, max_y])
ax.set_zlim([0, max_z])


def update_points(t, x, y, z, points):
  ax.clear()
  ax.set_xlim([0, max_x])
  ax.set_ylim([0, max_y])
  ax.set_zlim([0, max_z])
  pos = ask(url)
  i = 0
  for key in pos.keys():
    x_coord, y_coord, z_coord = pos[key]
    x[i] = x_coord
    y[i] = z_coord
    z[i] = y_coord
    ax.text(x_coord, z_coord, max_z - y_coord, PART_NAMES[i], color='red')
    ax.scatter(x_coord, z_coord, max_z - y_coord, 'ro')
    i += 1
    
  new_x = x 
  new_y = y 
  new_z = z 
  points.set_data(new_x, new_y)
  points.set_3d_properties(new_z, 'z')
  return points, 

ani = animation.FuncAnimation(fig, update_points, fargs=(x, y, z, points),
                               frames=1, interval=33)

plt.show()
