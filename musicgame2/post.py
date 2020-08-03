import requests

sess=requests.session()

url='https://game2.wmctf.wetolink.com:4432'

sess.get(url)

def tank_go(direct):
    files = {
        'upfile': open("{}.wav".format(direct), 'rb')
        }
    res=sess.post(url,files=files)
    


actions=[
    'up',
    'up',
    'left',
    'left',
    'down',
    'down',
    'down',
    'down',
    'right',
    'right',
    'right',
    'right',
    'up',
    'up',
    'right',
    'right',
    'down',
    'down',
    'down',
    'down',
    'right',
    'right',
    'up',
    'up',
    'right',
    'right',
    'right',
    'down',
    'down',
    'down',
    'down',
    'down',
    'down',
    'down',
    'down',
    'right',
    'right',
    'right',
    'right',
    'right',
    'right',
    'right',
    'right',
    'right',
]

for action in actions:
    tank_go(action)

res=sess.get(url)
print(res.text)