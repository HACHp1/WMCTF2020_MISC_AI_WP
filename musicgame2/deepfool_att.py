
from utils import *
from word2vec import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import copy

from tensorflow import keras
import librosa
import soundfile as sf

class_num = 4  # 总类别数


v_max_steps = 100
v_overshoot = 0.001 # 0.001 需要跨越目标的量（deepfool是刚好到达决策边界，需要越过边界）




I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK=True



def get_model():
    '''
    获取模型
    '''
    model = keras.models.load_model('model.h5')
    model.trainable = False  # 冻结模型参数
    # model.summary()
    return model


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


def loss_func(logits, I, k):
    return logits[0, I[k]]

def deepfool(sample, model, overshoot=0.02, max_iter=50):
    """
       :param sample: Image of size
       :param model: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool

       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed sample
    """

    f_image = model(sample).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:class_num]
    label = I[0]

    input_shape = np.shape(sample)
    pert_image = copy.deepcopy(sample)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = label

    if np.all(fs >= -.0001) and np.all(fs <= 1.0001):
                    # 检测是不是用的softmax前的一层（若和与1很接近，则很有可能是经过sm处理过）
                    if np.allclose(np.sum(fs, axis=1), 1.0, atol=1e-3):
                        if not I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")


    while k_i == label and loop_i < max_iter:

        pert = np.inf

        one_hot_label_0 = tf.one_hot(label, class_num)

        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)

            loss_value = loss_func(fs, I, 0)

        grad_orig = tape.gradient(loss_value, x)

        for k in range(1, class_num):
            one_hot_label_k = tf.one_hot(I[k], class_num)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                loss_value = loss_func(fs, I, k)

            cur_grad = tape.gradient(loss_value, x)

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / (np.linalg.norm(tf.reshape(w_k, [-1])))

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = sample + (1 + overshoot) * r_tot


        x = tf.Variable(pert_image)

        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1


    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image



if __name__ == '__main__':

    sr = 16000
    
    sample = get_wav_mfcc('example.wav').reshape(1,30,20)


    sample = tf.Variable(sample, dtype=tf.float32)
    model = get_model()

    # ----无目标攻击


    r_tot, iter_i, label, k_i, sample_ea = deepfool(
        sample, model, v_overshoot, v_max_steps)

    sample_ea=sample_ea.numpy()

    result = model.predict(sample_ea)

    print()
    print('deepfool攻击；真实：0',  '攻击后：', np.argmax(result), '迭代次数：', iter_i)


    sample=librosa.feature.inverse.mfcc_to_audio(sample_ea.numpy().reshape(30, 20).T)
    sf.write('left.wav' , sample,sr)
