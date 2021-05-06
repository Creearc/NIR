import maya.cmds as cmds
import requests

def ask(url):
  url = '{}/pos'.format(url)

  response = requests.post(url=url, timeout=10)
  out = response.json()['pos']

  return out

HOST = '192.168.68.101'
PORT = 56088

url = 'http://{}:{}'.format(HOST, PORT)

objs = cmds.ls('*Cube*', type='transform')

while True:
  pos = ask(url)
  for j in range(17):
    x, y, z = pos[j]
    cmds.setAttr(objs[j] + '.translateX', x)
    cmds.setAttr(objs[j] + '.translateY', y)
    cmds.setAttr(objs[j] + '.translateZ', z)
  cmds.setKeyframe(objs)



"""
print(objs)

f = open("A:/points/3.txt", 'r')
for frame in range(50):
    cmds.currentTime(frame * 60) 
    s = f.readline().split('|')
    for j in range(68):
        #print(objs[j])
        posX = int(s[j + 1].split()[0])
        posY = -int(s[j + 1].split()[1])
        cmds.setAttr(objs[j] + '.translateX', posX)
        cmds.setAttr(objs[j] + '.translateY', posY)
        cmds.setAttr(objs[j] + '.translateZ', 0)
    cmds.setKeyframe(objs)
"""
