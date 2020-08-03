from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import struct
import os


# 图片切割
def segment(im):
    wid = 28
    up = 0
    down = 128
    im_new = []

    for i in range(23):
        for j in range(32):
            im1 = im.crop((wid * j, wid * i, wid * (j + 1), wid * (i+1)))
            # im1 = im.crop((wid * i, up, wid * (i + 1), down))  # 分4段
            im_new.append(im1)
    return im_new


def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_enist_train(path, kind='train'):
    labels_path = os.path.join(
        path, 'emnist-letters-%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(
        path, 'emnist-letters-%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def np2str(np_image):
    tmp = ''
    for i in np_image:
        tmp += hex(int(i))[2:]
    return tmp


def img2str(image):
    tmp = ''
    np_img = np.array(image).reshape(-1)
    for i in np_img:
        tmp += hex(int(i))[2:]
    return tmp


# print(len(img2str(segment(myimg)[4])))
# print(len(np2str(images[0])))

img_dic={}

images, labels=load_mnist_train('data')
for i in range(images.shape[0]):
    img_dic[np2str(images[i])]=labels[i]

letter_li = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
             'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 'v', 'w', 'x', 'y', 'z']

images, labels = load_enist_train('data/gzip', kind='train')
# print(labels[:100])
# exit()
# Image.fromarray(np.transpose(images[100].reshape(28,28))).show()


for i in range(images.shape[0]):
    img_dic[
        np2str(
            np.transpose(images[i].reshape(28, 28)).reshape(-1)
        )
    ] = letter_li[labels[i]-1]


file_name = 'all.png'
myimgs = segment(Image.open(file_name))

for tmpimg in myimgs:
    print(
        img_dic[img2str(tmpimg)],
        end=''
    )


# tmpimg=segment()
# tmpimg[0].save('test.jpg', 'jpeg')  # 保存
