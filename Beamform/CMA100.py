import librosa
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Beamform.BFModel import beamforming_model
import Beamform.tools
import matplotlib.pyplot as plt


class CMA100():
    def __init__(self, num_angle=6, num_micarray=4, num_mic=5, r=[2 * i for i in [0.008, 0.016, 0.032, 0.064]], 
    fs=32000, L=1024, c=340, decay_angle=120):
        '''
        args:
            num_angle: 麦克风阵列分布的平面角度数
            num_micarray: 对于某一个角度theta, 该角度共有多少组mic阵列
            num_mic: 每组阵列有多少个mic
            r: 对于某一个角度theta, 不同组阵列的阵元间距, 排序从小到大(会影响到band_filter代码的编写)
            fs: 信号采样率
            L: 阵列快拍数
            c: 音速
            decay_angle: 抑制的俯仰角度数 
        '''
        assert len(r) == num_micarray, "对某个角度theta, 有多少组mic阵列就应该设置多少组半径!"

        self.mics_model_list = []
        for i in range(num_angle):
            self.mics_model_list.append([])
            theta = i * np.pi / num_angle
            for j in range(num_micarray):
                mic_x = [0.000]
                mic_y = [0.000]
                ri = r[j]
                for k in range((num_mic-1)//2):
                    mic_x.append((k+1)*ri * np.cos(theta))
                    mic_y.append((k+1)*ri * np.sin(theta))
                    mic_x.append((k+1)*(-ri) * np.cos(theta))
                    mic_y.append((k+1)*(-ri) * np.sin(theta))
                mic_z = [5.000 for num in range(len(mic_x))]
                mics_model = beamforming_model(mic_x, mic_y, mic_z, fs=fs, L=L, c=c)
                self.mics_model_list[i].append(mics_model)
        self.L = L

        # 信源数目及信源坐标
        self.num_signals = 0
        self.S_x = np.empty([0,])
        self.S_y = np.empty([0,])
        self.S_z = np.empty([0,])
        self.source_signals_rfft = np.zeros((5,self.L//2 + 1), dtype=complex) # 最多分析五个信源
        self.source_i = 0

        self.decay_angle = decay_angle
        self.enhanced_signal = np.zeros((1, self.L), dtype=complex) # 初始化增强后的信号
        self.r = r # 阵元间距
        self.angle = [np.pi/num_angle * i for i in range(num_angle)]
        self.c = c # 音速
        self.fs = fs

        # 设置基本频率以及 k值分配
        '''
        for example : fs = 16000, frameLength=512, k_value=[256, 128, 64, 32, 16]
        '''
        self.k_value = []
        k_max = self.L//2 # k索引的最大值
        for k in range(len(self.mics_model_list[0])):  #阵元间距排序从小到大， 频段从大到小
            if k == len(self.mics_model_list[0]) - 1:
                k_range = [0, k_max]
            else:
                k_range = [k_max//2 + 1, k_max]
            self.k_value.append(k_max)
            k_max = k_max//2
        
        self.f_unit = (self.fs//2) / self.k_value[0]
    
    def getCMA100(self):
        return self.mics_model_list

    # 重置信号信息, 用于流式处理。
    def reset(self):
        # 信源数目及信源坐标
        self.num_signals = 0
        self.S_x = np.empty([0,])
        self.S_y = np.empty([0,])
        self.S_z = np.empty([0,])
        self.source_signals_rfft = np.zeros((5,self.L//2 + 1), dtype=complex) # 最多分析五个信源
        self.source_i = 0

        for i in range(len(self.mics_model_list)):
            for j in range(len(self.mics_model_list[i])):
                self.mics_model_list[i][j].reset()

    def add_signal(self, S_x, S_y, S_z, signal_rfft=None):
        self.S_x = np.concatenate((self.S_x, np.array([S_x])),axis=0)
        self.S_y = np.concatenate((self.S_y, np.array([S_y])),axis=0)
        self.S_z = np.concatenate((self.S_z, np.array([S_z])),axis=0)
        assert self.S_x.shape == self.S_y.shape == self.S_z.shape, "Fault: x, y, z shape don't equal"
        assert signal_rfft.shape[0] == self.L//2 + 1, "signal.shape should be equal to self.L!"
        self.num_signals += 1

        # 更新CMA100所记录的信号源
        self.source_signals_rfft[ self.source_i, :] = signal_rfft
        self.source_i += 1

        # 使得每一个阵列接受新的信号源:
        for i in range(len(self.mics_model_list)):
            for j in range(len(self.mics_model_list[i])):
                self.mics_model_list[i][j].add_signal(S_x, S_y, S_z, signal_rfft)
    
    def set_signal(self, i, j, m, signal_rfft=None):
        assert signal_rfft.shape[0] == self.L//2 + 1, "signal.shape should be equal to self.L!"
        self.num_signals += 1
        # 更新CMA100所记录的信号源
        self.source_signals_rfft[ self.source_i, :] = signal_rfft
        self.source_i += 1
        self.mics_model_list[i][j].set_signal(m, signal_rfft) # 第i个角度， 第j组半径阵列， 导入信号singal_rfft


        

    
    # 不同阵元间距 的 阵列 处理 不同频段的信号, 将不同阵列接受矩阵的部分k值置0, 并返回k的临界点
    '''
    def __band_filter(self):
    '''
    '''
        以CMA100为例, 采样率32k, 帧长1024, FFT后是513个频点。 
        5组阵元间距: [0.008, 0.016, 0.032, 0.064, 0.128], 间距越短， 处理的频率越低
        513个频点对应的是0-16k的频率, (根据Nyquist采样定理, 一半的频率即可无失真重建目标频率信号)
        因此有
        阵元间距d=0.008对应的频段： 0-1k , 对应FFT的k值: [0, 32]   ->>>> 2^0 - 1 , 2^5
        d = 0.016对应的频段: 1k-2k 对应FFT的k值: [33, 64]   ->>>>   2^5 + 1 , 2^6
        d = 0.032对应的频段: 2k-4k 对应FFT的k值: [65, 128]  ->>>> 2^6 + 1 , 2^7
        d = 0.064对应的频段: 4k-8k 对应FFT的k值: [129, 256] ->>>> 2^7 + 1 , 2^8
        d = 0.128对应的频段: 8k-16k 对应FFT的k值: [257, 512] ->>>> 2^8 + 1 , 2^9
        '''
    '''
        k_value = []
        for i in range(len(self.mics_model_list)):
            k_max = self.L//2 # k的索引最大值
            for j in range(len(self.mics_model_list[i])): # 阵元间距排序从小到大， 频段从大到小
                if j == len(self.mics_model_list[i])-1: # 对阵元间距最大的阵列组合处理
                    k_range = [0, k_max]
                else:
                    k_range = [k_max//2 + 1, k_max]
                self.mics_model_list[i][j].reserveValueOf_Xk(k_range=k_range)
                k_value.append(k_max)
                k_max = k_max//2
        
        return k_value
    '''

    def band_filter(self):
        # 返回值k_value: 离散频率k分段， 如： [512,256,128,64,32]
        # 返回值f_value: 信号频率f分段,  如： [16000, ..., 0]
        k_value = []
        k_max = self.L//2 # k索引的最大值
        for k in range(len(self.mics_model_list[0])):  #阵元间距排序从小到大， 频段从大到小
            if k == len(self.mics_model_list[0]) - 1:
                k_range = [0, k_max]
            else:
                k_range = [k_max//2 + 1, k_max]
            for theta_i in range(len(self.mics_model_list)): # 遍历每一个角度， 对每个角度的阵列组合进行带通滤波
                self.mics_model_list[theta_i][k].reserveValueOf_Xk(k_range=k_range)
            k_value.append(k_max)
            k_max = k_max//2
        
        f_unit = (self.fs//2) / k_value[0] # fft后每个k对应的频率
        return k_value, f_unit

    # 抑制掉超过俯仰角的信号频率成分
    def angle_filter(self, decay_angle = None):
        # decay_angle 是需要抑制信号的俯仰到达角度。
        if decay_angle == None: decay_angle = self.decay_angle
        decay_angle = decay_angle/180 * np.pi / 2
        k_value, f_unit = self.band_filter() # [512, 256, 128, 64, 32],   32000//2 /512
        
        #依次对每一个频段进行遍历检查， 处理阵元半径从小到大， 处理频段从大到小
        
        for i in range(len(self.k_value)):
            d = self.r[i] # 获取阵元间距

            #n0 = d*np.sin(decay_angle)/self.c * f_value[i] # 获取在俯仰角之外， 阵元间距导致的数组位移量
            # 获取该阵元间距下， 对应的频段k的取值范围。
            # 以标准的CMA100为例： 32k采样率，帧长1024，FFT点数513（ [0, 512]）。 子阵列阵元r=0.008处理257-512的k索引数。
            k_range = [self.k_value[i+1]+1, self.k_value[i]] if i != len(k_value)-1 else [0, self.k_value[-1]]
            for k in range(k_range[0], k_range[1]+1, 1):
                n0 = d * np.sin(decay_angle) / self.c * self.fs # 统计对于某一个频点， 该频点对应的到达的波程差数
                phase_delta = np.exp(1j * 2 * np.pi * k * n0/self.L) # 计算对于特定频点造成数组位移量所需要的相位角度差。 Xk = |Xk|exp(j*phi)
                angle = np.angle(phase_delta)

                for j in range(len(self.angle)): # 遍历每组阵列角度
                    if np.abs(self.mics_model_list[j][i].getPhaseXk_delta(k)) > angle: # 如果相位偏移量大于阈值
                        for jj in range(len(self.angle)): # 则对于处理该频点的每一组阵列模型，我们要抑制该频点的能量
                            self.mics_model_list[jj][i].decayValueOf_Xk(k, factor=0)
    
    def angleFilter_doa(self, decay_angle = None, bf_method = "cbf"):
        # decay_angle 是需要抑制信号的俯仰到达角度。
        if decay_angle == None: decay_angle = self.decay_angle
        decay_angle = decay_angle/180 * np.pi / 2

        # 确定搜索角时的P矩阵
        P = np.zeros((257, 87, 31, len(self.angle))) # (k, rou_range.shape, theta.shape)

        # 俯仰角抑制 + 声源定位
        for i in range(len(self.k_value)):
            d = self.r[i]
            k_range = [self.k_value[i+1]+1, self.k_value[i]] if i != len(self.k_value)-1 else [0, self.k_value[-1]]
            for k in range(k_range[0], k_range[1]+1, 1):
                n0 = d * np.sin(decay_angle) / self.c * (self.fs)
                phase_delta = np.exp(1j * 2 * np.pi * k * n0/self.L)
                angle = np.abs(np.angle(phase_delta))
                # 遍历每一个角度
                # 执行俯仰角抑制算法
                # 在抑制频点能量后

                # 俯仰角处理后， 执行到达角确定算法
                # 确定到达角后， 找到处理该角度的mic阵列，用该阵列做俯仰角估计，确定声源空间方位
                # 根据声源空间方位执行 CBF / MVDR 算法
                phase_angle_array = np.array([])
                for j in range(len(self.angle)):
                    phase_angle = self.mics_model_list[j][i].getPhaseXk_delta(k)
                    # 该频点俯仰角不对劲， 将其抑制后跳出角度遍历，处理下一个频点, 也就不做beamforming等操作了
                    if np.abs(phase_angle) > angle:
                        for jj in range(len(self.angle)):
                            self.mics_model_list[jj][i].decayValueOf_Xk(k, factor=0.0)
                        phase_angle_array = np.array([])
                        break
                    # 如果该频点俯仰角在观测区域内， 我们需要统计下该频点能量， 频点能量最大的阵列角度为到达角
                    else:
                        phase_angle_array = np.append(phase_angle_array, phase_angle)
                    
                # 确定声源到达角: 如果信号的该频点在关注范围外， 则直接处理下一个频点
                if phase_angle_array.shape[0] == 0:
                    continue
                else:
                    angle_index = np.argmax(np.abs(phase_angle_array)) # 找到相角差最大的索引
                    #phase_angle = phase_angle_array[angle_index] # 相位差角
                    theta = self.angle[angle_index] if phase_angle >=0 else -self.angle[angle_index] # 确定平面到达角粗略角度，准备搜索，[-pi, pi]
                    theta_delta = 15/180*np.pi
                    rou_range = np.arange(0, (5.0 - 0) * np.tan(60/180 * np.pi), 0.10)
                    theta_range = np.arange(theta - theta_delta, theta + theta_delta+1e-6, 1/180 * np.pi)
                    if bf_method == "cbf":
                        P[k, :, :, angle_index] = self.doa_v1(k, theta_range, rou_range)
                    if bf_method == "mvdr":
                        P[k, :, :, angle_index] = self.doa_v2(k, theta_range, rou_range)
                    if bf_method == "music":
                        P[k, :, :, angle_index] = self.doa_v3(k, theta_range, rou_range)
        
        P_fullband = np.sum(P, axis=0, keepdims=False)
        #print("P_fullband is", P_fullband)

        # 找到P_fullband的极值点， 计算出声源位置
        P, index_i, index_j, index_k = 0, 0, 0, 0
        for i in range(P_fullband.shape[0]):
            for j in range(P_fullband.shape[1]):
                for k in range(P_fullband.shape[2]):
                    if P < np.abs(P_fullband[i][j][k]):
                        P = np.abs(P_fullband[i][j][k])
                        index_i = i
                        index_j = j
                        index_k = k
        theta = self.angle[index_k] - theta_delta + index_j * 1/180 * np.pi
        rou = 0.10 * index_i
        x = rou * np.cos(theta)
        y = rou * np.sin(theta)
        print("x is", x)
        print("y is", y)

    # 将阵列信号波束形成到空间定点位置。
    def beamforming(self, x, y, z):
        # 遍历所有mic子阵列
        for i in range(len(self.k_value)):
            for j in range(len(self.angle)):
                # 将所有子阵列进行波束形成
                self.mics_model_list[j][i].beamform(x, y, z)


    # music方法
    def doa_v3(self, k, theta_range, rou_range):
        # doa, 求信号功率
        z = 0
        i=0
        if k <= 128 : i=1
        if k <= 64 : i=2
        if k <= 32 : i=3
        signal_matrix = np.zeros((10, 1), dtype=complex)
        micx_matrix = np.zeros((10, 1))
        micy_matrix = np.zeros((10, 1))
        micz_matrix = np.zeros((10, 1))
        #print(self.mics_model_list[0][i].receive_signals_rfft[:,k].shape)
        signal_matrix[:5,:] = self.mics_model_list[0][i].receive_signals_rfft[:,k:(k+1)] # shape: [M,1]
        signal_matrix[5:10, :] = self.mics_model_list[3][i].receive_signals_rfft[:,k:(k+1)]

        covariance = np.dot(signal_matrix, np.conj(signal_matrix.T))
        # 计算特征值和特征向量
        lamb, E = np.linalg.eig(covariance)
        # 取噪声子空间
        idx = np.argsort(lamb)[::-1]  # 从大到小排序
        E = E[:, idx]
        En = E[:, 1:] # 我们只认为有一个信号源，剩下的都是噪声子空间

        

        micx_matrix[:5, :] = self.mics_model_list[0][i].mic_x # shape: [M, 1]
        micx_matrix[5:10, :] = self.mics_model_list[3][i].mic_x
        micy_matrix[:5, :] = self.mics_model_list[0][i].mic_y
        micy_matrix[5:10, :] = self.mics_model_list[3][i].mic_y
        micz_matrix[:5, :] = self.mics_model_list[0][i].mic_z
        micz_matrix[5:10, :] = self.mics_model_list[3][i].mic_z

        P = np.zeros((rou_range.shape[0], theta_range.shape[0]))
        for i in range(rou_range.shape[0]):
            for j in range(theta_range.shape[0]):
                x = rou_range[i] * np.cos(theta_range[j])
                y = rou_range[i] * np.sin(theta_range[j])
                distance_matrix = np.sqrt( (x - micx_matrix)**2 + (y - micy_matrix)**2 + (z-micz_matrix)**2 )
                phase_matrix = np.exp(1j*2*np.pi*k*self.fs*distance_matrix/self.c/self.L)
                phase_matrix_H = np.conj(phase_matrix.T)
                P[i,j] = 1 / np.abs( np.dot(    np.dot(phase_matrix_H, En) , np.dot(np.conj(En).T, phase_matrix)    ))
        return P

        
    
    # mvdr方法
    def doa_v2(self, k, theta_range, rou_range):
        # doa, 求信号功率
        z = 0
        i=0
        if k <= 128 : i=1
        if k <= 64 : i=2
        if k <= 32 : i=3
        signal_matrix = np.zeros((10, 1), dtype=complex)
        micx_matrix = np.zeros((10, 1))
        micy_matrix = np.zeros((10, 1))
        micz_matrix = np.zeros((10, 1))
        #print(self.mics_model_list[0][i].receive_signals_rfft[:,k].shape)
        signal_matrix[:5,:] = self.mics_model_list[0][i].receive_signals_rfft[:,k:(k+1)] # shape: [M,1]
        signal_matrix[5:10, :] = self.mics_model_list[3][i].receive_signals_rfft[:,k:(k+1)]

        covariance = np.dot(signal_matrix, np.conj(signal_matrix.T))
        covariance_inverse = np.linalg.pinv(covariance) # 如果使用逆， 则使用np.linalg.inv

        micx_matrix[:5, :] = self.mics_model_list[0][i].mic_x # shape: [M, 1]
        micx_matrix[5:10, :] = self.mics_model_list[3][i].mic_x
        micy_matrix[:5, :] = self.mics_model_list[0][i].mic_y
        micy_matrix[5:10, :] = self.mics_model_list[3][i].mic_y
        micz_matrix[:5, :] = self.mics_model_list[0][i].mic_z
        micz_matrix[5:10, :] = self.mics_model_list[3][i].mic_z

        P = np.zeros((rou_range.shape[0], theta_range.shape[0]))
        for i in range(rou_range.shape[0]):
            for j in range(theta_range.shape[0]):
                x = rou_range[i] * np.cos(theta_range[j])
                y = rou_range[i] * np.sin(theta_range[j])
                distance_matrix = np.sqrt( (x - micx_matrix)**2 + (y - micy_matrix)**2 + (z-micz_matrix)**2 )
                phase_matrix = np.exp(1j*2*np.pi*k*self.fs*distance_matrix/self.c/self.L)
                phase_matrix_H = np.conj(phase_matrix.T)
                P[i,j] = 1 / np.abs( np.dot( phase_matrix_H, np.dot(covariance_inverse, phase_matrix) ) )
        
        return P

        

    # cbf方法
    def doa_v1(self, k, theta_range, rou_range):
        '''
        到达角搜索， 在 [phi - phi_delta, phi+phi_delta]内对声源到达角进行搜索
        确定声源到达角和俯仰角度后， 进行 cbf / mvdr

        算法: 
            1. 首先根据k值确定我们需要beamforming哪个半径组的麦阵, 根据k_value的区间。
            2. 确定麦形， 这里选择十字形， 分别是第0个角度和第4个角度的麦阵
            3. 通过1, 2确定子阵列, 在给定声源到达角中进行theta平面到达角搜索, 返回theta
        '''
        # doa, 求信号功率
        z = 0
        i=0
        if k <= 128 : i=1
        if k <= 64 : i=2
        if k <= 32 : i=3
        signal_matrix = np.zeros((10, 1), dtype=complex)
        micx_matrix = np.zeros((10, 1))
        micy_matrix = np.zeros((10, 1))
        micz_matrix = np.zeros((10, 1))
        #print(self.mics_model_list[0][i].receive_signals_rfft[:,k].shape)
        signal_matrix[:5,:] = self.mics_model_list[0][i].receive_signals_rfft[:,k:(k+1)] # shape: [M,1]
        signal_matrix[5:10, :] = self.mics_model_list[3][i].receive_signals_rfft[:,k:(k+1)]

        covariance = np.dot(signal_matrix, np.conj(signal_matrix.T))

        micx_matrix[:5, :] = self.mics_model_list[0][i].mic_x # shape: [M, 1]
        micx_matrix[5:10, :] = self.mics_model_list[3][i].mic_x
        micy_matrix[:5, :] = self.mics_model_list[0][i].mic_y
        micy_matrix[5:10, :] = self.mics_model_list[3][i].mic_y
        micz_matrix[:5, :] = self.mics_model_list[0][i].mic_z
        micz_matrix[5:10, :] = self.mics_model_list[3][i].mic_z

        P = np.zeros((rou_range.shape[0], theta_range.shape[0]))
        for i in range(rou_range.shape[0]):
            for j in range(theta_range.shape[0]):
                x = rou_range[i] * np.cos(theta_range[j])
                y = rou_range[i] * np.sin(theta_range[j])

                distance_matrix = np.sqrt( (x - micx_matrix)**2 + (y - micy_matrix)**2 + (z-micz_matrix)**2 )
                phase_matrix = np.exp(1j*2*np.pi*k*self.fs*distance_matrix/self.c/self.L)
                phase_matrix_H = np.conj(phase_matrix.T)
                P[i,j] = np.abs(np.dot( phase_matrix_H, np.dot(covariance, phase_matrix) ))
        
        return P
        
        

    def build_signal_rfft(self, method=0):
        if method == 0:
            signal_rfft = np.zeros(self.L // 2 + 1, dtype=complex)
            for i in range(len(self.angle)):
                for j in range(len(self.r)):
                    signal_rfft += np.sum(self.mics_model_list[i][j].receive_signals_rfft, axis = 0)
            return signal_rfft
    
    # check 一下相位协方差矩阵
    def check(self):
        # mic编号：   4 2 0 1 3
        xk1 = self.mics_model_list[0][0].receive_signals_rfft[4,145]
        xk2 = self.mics_model_list[0][0].receive_signals_rfft[2,145]
        xk3 = self.mics_model_list[0][0].receive_signals_rfft[0,145]
        xk4 = self.mics_model_list[0][0].receive_signals_rfft[1,145]
        xk5 = self.mics_model_list[0][0].receive_signals_rfft[3,145]

        print(np.angle([xk1, xk2, xk3, xk4, xk5]))
        print(np.angle(xk5) - np.angle(xk4))
        print(np.angle(xk4) - np.angle(xk3))


            

if __name__ == "__main__":
    CMA = CMA100()
    a = CMA.band_filter()
    print()


                