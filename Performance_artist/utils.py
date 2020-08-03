from PIL import Image
import numpy as np
# filename = 'data/test_split/'
def load_data(filename, imgarr,datalen=50000):
    # print(datalen)
    for i in range(datalen):
        for j in range(4):
            im1 = Image.open(filename + str(i).rjust(5, '0') + str(j) + '.jpg')
            char1 = im1.getdata()
            char1 = np.array(char1)
            char1.reshape((30, 43))
            imgarr.append(char1)


