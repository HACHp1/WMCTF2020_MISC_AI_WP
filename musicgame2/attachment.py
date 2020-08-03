import numpy as np
import os
import librosa, librosa.display

from tensorflow import keras

def get_wav_mfcc(wav_path):
    y, sr = librosa.load(wav_path,sr=None)
    data=librosa.feature.mfcc(y,sr=sr)
    data=np.array(data)

    '''to unified format'''
    while len(data[0])>30:
        data=np.delete(data,-1,axis=1)
        data=np.delete(data,0,axis=1)
    while len(data[0])<30:
        data=np.insert(data,-1,values=data.T[-1],axis=1)
    return data.T

def checkdifferent(path):
    mfcc1=get_wav_mfcc('example.wav')
    mfcc2=get_wav_mfcc(path)

    print(np.mean(np.abs(mfcc1-mfcc2)))
    return True

    if np.mean(np.abs(mfcc1-mfcc2))<4:
        return True
    else:
        return False

model = keras.models.load_model('model.h5')
def detect(path):
    ret=model.predict(get_wav_mfcc(path).reshape(1,30,20))
    return (ret.max(),ret.argmax())

if __name__ == "__main__":
    # path='yours.wav'
    path='example.wav'
    # path='example.wav'
    # try:
    if not checkdifferent(path):
            print('sorry,it is too different with example.wav')
            exit()
    (num,lable)=detect(path)
    # except:
    #     print('ERROR')
    #     exit()
    if num<0.9:
            print("I can't detect with certainty")
            exit() 
    if lable==0:
        print('up')
    elif lable==1:
        print('left')
    elif lable==2:
        print('down')
    elif lable==3:
        print('right')
    print(num)
