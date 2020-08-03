'''
Carlini_Wagner
'''

from attack_lib.cw import *


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow import keras
import copy
import librosa
import soundfile as sf
import time

class_num = 4  # 总类别数

target_label = 1  # 目标类别


def get_wav_mfcc(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    data = librosa.feature.mfcc(y, sr=sr)
    data = np.array(data)

    '''to unified format'''
    while len(data[0]) > 30:
        data = np.delete(data, -1, axis=1)
        data = np.delete(data, 0, axis=1)
    while len(data[0]) < 30:
        data = np.insert(data, -1, values=data.T[-1], axis=1)
    return data.T


def get_model():
    '''
    获取模型
    '''
    model = keras.models.load_model('model.h5')
    model.trainable = False  # 冻结模型参数
    # model.summary()
    return model



if __name__ == '__main__':
    sr = 16000
    
    sample = get_wav_mfcc('example.wav').reshape(1,30,20)
    

    sample = tf.Variable(sample, dtype=tf.float32)
    model = get_model()
    # model.summary()
    # exit()
    # ----有目标攻击

    # 注意：cw需要softmax层之前的输出！！！


    attack = CarliniL2(
        model,sample_shape=(1,30,20),
        boxmax=1,boxmin=-1
        )

    attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK=True

    sample = tf.cast(sample, tf.float32)

    inputs = sample  

    print(inputs.shape)

    targets = np.eye(class_num)[target_label]
    print(targets)

    list_targets = []
    list_targets.append(targets)

    t0=time.time()
    sample_ea = attack.attack(inputs, list_targets)
    t1=time.time()

    result = model.predict(sample_ea)

    print()
    print('c&w攻击；真实：0', '攻击后：', np.argmax(result))
    print('花费时间：{} 秒'.format(str(t1-t0)))

    sample=librosa.feature.inverse.mfcc_to_audio(sample_ea.numpy().reshape(30, 20).T)
    sf.write('left.wav' , sample,sr)

    
