'''
当迭代次数为1时，是FGM攻击；不为1时则为修改版迭代攻击
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow import keras
import librosa
import soundfile as sf

class_num = 4  # 总类别数

target_label = 3  # 目标类别


v_max_steps = 10  # 当迭代次数为1时，是FGSM攻击；否则为IFGSM攻击。
v_step_alpha = 20


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

# def get_wav_mfcc(wav_path):
#     y, sr = librosa.load(wav_path, sr=None)
#     print(sr)
#     data = librosa.feature.mfcc(y, sr=sr)
#     data = np.array(data)

#     return data.T


def get_model():
    '''
    获取模型
    '''
    model = keras.models.load_model('model.h5')
    model.trainable = False  # 冻结模型参数
    # model.summary()
    return model


def loss_object(label, predict):
    return tf.keras.losses.categorical_crossentropy(label, predict)


def train_step(model, sample, label):
    '''
    计算梯度
    '''
    with tf.GradientTape() as tape:
        tape.watch(sample)
        predict = model(sample)
        loss = loss_object(label, predict)
    grad = tape.gradient(loss, sample)
    normed_grad = grad/tf.reduce_sum(tf.square(grad))

    return normed_grad


def target_attack(sample, model, target_label, max_steps, step_alpha):
    '''
    有目标梯度下降攻击，达到目标或者最大迭代值时停止

    注：此时的对抗样本没有进行可行域压缩

    返回：对抗样本, 一共进行的迭代次数i
    '''

    target_label = utils.to_categorical(target_label, class_num)

    for i in range(max_steps):
        signed_grad = train_step(model, sample, target_label)
        normed_grad = step_alpha * signed_grad
        sample = sample - normed_grad  # 有目标攻击时，梯度下降

        if np.argmax(target_label) == np.argmax(model(sample)):
            break

    return sample, i


def non_target_attack(sample, model, max_steps, step_alpha):
    '''
    无目标梯度下降攻击，达到目标或者最大迭代值时停止

    注：此时的对抗样本没有进行可行域压缩

    返回：对抗样本, 一共进行的迭代次数i
    '''
    target_label = np.argmax(model.predict(
        sample.numpy().reshape(1, 30, 20)))  # 先转化为numpy，否则会考虑batch_size而报错
    target_label = utils.to_categorical(target_label, class_num)

    for i in range(max_steps):
        signed_grad = train_step(model, sample, target_label)
        normed_grad = step_alpha * signed_grad
        sample = sample + normed_grad  # 无目标攻击时，梯度上升

        if np.argmax(target_label) != np.argmax(model(sample)):
            break

    return sample, i



if __name__ == '__main__':

    # sr = 16000
    sr=22050 

    sample = get_wav_mfcc('example.wav').reshape(1,30,20)
    
    # sample = sample.reshape(30,20).T
    # sample=librosa.feature.inverse.mfcc_to_audio(sample)
    # sf.write('left.wav' , sample,sr)
    # exit()
    
    sample = tf.Variable(sample, dtype=tf.float32)
    model = get_model()

    # ----有目标攻击
    sample_ea, iter_i = target_attack(
        sample, model, target_label, v_max_steps, v_step_alpha)

    result = model.predict(sample_ea)

    print()
    print('fgm攻击；真实：0', '攻击后：', np.argmax(result), '迭代次数：', iter_i+1)

    sample=librosa.feature.inverse.mfcc_to_audio(sample_ea.numpy().reshape(30, 20).T)
    sf.write('right.wav' , sample,sr)

    # ----无目标攻击
    # sample_ea, iter_i = non_target_attack(
    #     sample, model, v_max_steps, v_step_alpha)

    # result = model.predict(sample_ea)

    # print()
    # print('无目标攻击；真实：0', '攻击后：', np.argmax(result), '迭代次数：', iter_i+1)
    # sample=librosa.feature.inverse.mfcc_to_audio(sample_ea.numpy().reshape(30, 20).T)
    # sf.write('left.wav' , sample,sr)
