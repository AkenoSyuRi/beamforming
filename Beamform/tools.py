# 本文件用于编写处理语音信号的工具箱

import librosa
import os
import numpy as np

def CalRms(AudioData):
    # input: shape:[N,]
    # return: rms
    rms = np.sqrt(np.mean(AudioData ** 2))
    return rms

def ConvertToTargetDb(AudioData, TargetDb):
    # 将语音信号能量转化到TargetDb
    rms = CalRms(AudioData)
    scalar = 10 ** (TargetDb / 20) / (rms + 1e-7)
    AudioData *= scalar
    return AudioData

def is_clipped(AudioData):
    # 判断语音信号是否存在clip现象
    return any(np.abs(AudioData) > 0.999)

def NormAmplitude(AudioData):
    # 用于语音信号归一化
    max = np.max(np.abs(AudioData))+0.001
    return AudioData/max

def NormLength(audio, num_samples=16000*5):
    # 截取或者pad音频数组到指定长度
    if audio.shape[0] < num_samples:
        shortage = num_samples - audio.shape[0]
        audio = np.pad(audio, (0, shortage), mode='wrap')
    else:
        audio = audio[:num_samples]
    return audio

def mixAudios(audio1, audio2, num_samples=16000*5):
    return NormLength(audio1, num_samples) + NormLength(audio2, num_samples)
    

    




# -----------------坐标系工具---------------------

def Convert_Xyz2Spherical(x, y, z):
    # 转换x, y, z到球坐标系下
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y / x)
    phi = np.arccos(r / z)
    return r, theta, phi

def Convert_Spherical2Xyz(r, theta, phi):
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

def CalDistanceInSpherical(axis1:list, axis2:list):
    # 在球坐标系下计算两点的距离
    rou_1, theta_1, phi_1 = axis1[0], axis1[1], axis1[2]
    rou_2, theta_2, phi_2 = axis2[0], axis2[1], axis2[2]
    x_1, y_1, z_1 = Convert_Spherical2Xyz(rou_1, theta_1, phi_1)
    x_2, y_2, z_2 = Convert_Spherical2Xyz(rou_2, theta_2, phi_2)
    return np.sqrt( (x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2 )



