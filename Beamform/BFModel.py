import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Beamform.tools import is_clipped, NormAmplitude

'''
Author: lizhinan
Date: 23/04/20
'''

class beamforming_model():
    def __init__(self, mic_x:list, mic_y:list, mic_z:list, fs=32000, L=1024, c=340, ref_x=None, ref_y=None, ref_z =None):
        self.mic_x = np.array(mic_x).reshape(-1, 1) # shape: [M, 1]
        self.mic_y = np.array(mic_y).reshape(-1, 1)
        self.mic_z = np.array(mic_z).reshape(-1, 1)
        self.ref_x = ref_x if ref_x != None else np.mean(mic_x)
        self.ref_y = ref_y if ref_y != None else np.mean(mic_y)
        self.ref_z = ref_z if ref_y != None else np.mean(mic_z)
        assert self.mic_x.shape == self.mic_y.shape == self.mic_z.shape, "Fault: coordirate wrong! "
        self.M = self.mic_x.shape[0] # 阵元数目
        self.c = c # 音速
        self.fs = fs

        # self.T = T # 信号持续的时间

        self.L = L # 信号采样点数 (快拍数)

        # 信源数目及信源坐标
        self.num_signals = 0
        self.S_x = np.empty([0,])
        self.S_y = np.empty([0,])
        self.S_z = np.empty([0,])
        self.source_signals_rfft = np.zeros((5,self.L//2 + 1), dtype=complex) # 最多分析五个信源
        self.source_i = 0

        # 各个阵元到参考阵元之间的距离
        self.distances_matrix = np.sqrt( (self.ref_x - self.mic_x)**2 + (self.ref_y - self.mic_y)**2 + (self.ref_z - self.mic_z)**2 ) # shape:[M, 1]

        # 阵列接受到的信号
        self.receive_signals_rfft = np.zeros((self.M, self.L//2 + 1), dtype=complex)
        # 阵列接受信号的协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1) 
        # 参考mic/天线 接受到的信号
        self.ref_signals_rfft = np.zeros((1, self.L//2 + 1), dtype=complex)
        self.k_min = 0
        self.k_max = self.L//2
        self.f_unit = self.fs/2  / self.k_max
    
    def reset(self):
        # 重置信号信息

        # 信源数目及信源坐标
        self.num_signals = 0
        self.S_x = np.empty([0,])
        self.S_y = np.empty([0,])
        self.S_z = np.empty([0,])
        self.source_signals_rfft = np.zeros((5,self.L//2 + 1), dtype=complex) # 最多分析五个信源
        self.source_i = 0

        # 各个阵元到参考阵元之间的距离
        self.distances_matrix = np.sqrt( (self.ref_x - self.mic_x)**2 + (self.ref_y - self.mic_y)**2 + (self.ref_z - self.mic_z)**2 ) # shape:[M, 1]

        # 阵列接受到的信号
        self.receive_signals_rfft = np.zeros((self.M, self.L//2 + 1), dtype=complex)
        # 阵列接受信号的协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1) 
        # 参考mic/天线 接受到的信号
        self.ref_signals_rfft = np.zeros((1, self.L//2 + 1), dtype=complex)
        self.k_min = 0
        self.k_max = self.L//2

    def reserveValueOf_Xk(self, k_range):
        # 取阵列信号的频段， 在k_range以外的频段我们会将其置0
        self.k_min = k_range[0]
        self.k_max = k_range[1]
        assert self.k_min >= 0, "k的索引值应当大于等于0"
        assert self.k_max <= self.L//2 + 1, "k的索引值应当小于等于 FFT点数"
        for k in range(self.k_min):
            self.ref_signals_rfft[0, k] = 0
            for i in range(self.M):
                self.receive_signals_rfft[i, k] = 0
        
        for k in range(self.k_max+1, self.L//2+1, 1):
            self.ref_signals_rfft[0, k] = 0
            for i in range(self.M):
                self.receive_signals_rfft[i, k] = 0

        # 更新阵列接受信号的协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1)
    
    def getEnergy(self, k):
        # 返回阵列mic能量
        energy = 0.0
        for i in range(self.M):
            energy += np.abs(self.receive_signals_rfft[i, k]) ** 2
        return energy

    def getPhaseXk_delta(self, k):
        # 返回第k个频点的阵元相位角差
        
        angle = np.angle(self.receive_signals_rfft[:,k])
        # 因为相位角有周期性， 这里要分类讨论：
        if angle[0] >=0 and angle[1] >= 0:
            return np.abs(angle[0] - angle[1])
        if angle[0] <0 and angle[1] <0:
            return np.abs(angle[0] - angle[1])
        
        if angle[0] < 0 and angle[1] >= 0:
            # 交换angle[0], angle[1]交给下个判断处理
            swap = angle[0]
            angle[0] = angle[1]
            angle[1] = swap

        if angle[0] >=0 and angle[1] < 0:
            delta_angle1 = np.pi-angle[0] + (angle[1] + np.pi)
            delta_angle2 = angle[0] - angle[1]
            return min([delta_angle1, delta_angle2])
        
        # 相角归一化到[0, 2*pi]
        '''
        angle = np.angle(self.receive_signals_rfft[:,k])
        if angle[0] < 0 : angle[0] += (2*np.pi)
        if angle[1] < 0 : angle[1] += (2*np.pi)
        '''
        # return angle[0] - angle[1]

    def decayValueOf_Xk(self, k, factor=0.1):
        # 衰减对应频点的能量:
        for i in range(self.M):
            self.receive_signals_rfft[i, k] *= factor
            self.ref_signals_rfft[0, k] *= factor
        # 更新阵列协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1) 
        
        
    
    def add_signal(self, S_x, S_y, S_z, signal_rfft=None):
        self.S_x = np.concatenate((self.S_x, np.array([S_x])),axis=0)
        self.S_y = np.concatenate((self.S_y, np.array([S_y])),axis=0)
        self.S_z = np.concatenate((self.S_z, np.array([S_z])),axis=0)
        assert self.S_x.shape == self.S_y.shape == self.S_z.shape, "Fault: x, y, z shape don't equal"
        assert signal_rfft.shape[0] == self.L//2 + 1, "signal.shape should be equal to self.L!"
        self.num_signals += 1

        # 更新model所记录的信号源
        self.source_signals_rfft[ self.source_i, :] = signal_rfft
        self.source_i += 1        
        # 计算参考阵元与信号源之间的距离, 计算在fs采样率下的偏移
        ref_distance = np.sqrt( (S_x - self.ref_x)**2 + (S_y - self.ref_y)**2 + (S_z - self.ref_z)**2 )

        #n0_ref = ref_distance / self.c * self.fs

        # 计算不同频率的相位偏移， 生成参考阵元接收的频谱:
        ref_signal_fft = np.zeros(( self.L//2 + 1, ), dtype=complex)
        for k in range(self.L//2 + 1):
            # 先求指定频率的波长
            #k_lambda = self.c / (self.f_unit * k)
            # 再求传播距离等效于多少个波长
            #k_lambda_num = array_distance / k_lambda
            delta_t = ref_distance / self.c
            delta_n0 = delta_t * self.fs # 指定频率的移位序列

            # 再求参考阵元的接受信号:
            #ref_signal_fft[k] = signal_rfft[k] * np.exp(1j * 2 * np.pi * k_lambda_num) / ref_distance
            ref_signal_fft[k] = signal_rfft[k] * np.exp(1j * 2 * np.pi * k * delta_n0 / self.L) / ref_distance # 接受的幅度与距离成反比

        self.ref_signals_rfft[0,:] = self.ref_signals_rfft[0,:] + ref_signal_fft # 更新参考阵元信号

        # 计算各个阵元与信号源之间的距离
        array_distance = np.sqrt( (S_x - self.mic_x)**2 + (S_y - self.mic_y)**2 + (S_z - self.mic_z)**2 ) #shape: [M, 1]
        
        # 计算信号程差矢量和数组偏移
        #n0_i = array_distance / self.c * self.fs# shape: [M, 1]
        #print("n0_i is :", n0_i)
        #print("我是一组阵列,\n我的mic_x = {}\n我的mic_y = {}\n".format(self.mic_x, self.mic_y))
        #print("阵元与中心阵元信号位移数差值为：", n0_ref - n0_i)

        # 计算不同频率的相位偏移， 生成阵列的接收频谱:
        for k in range(self.L//2 + 1):
            delta_t = array_distance / self.c
            delta_n0 = delta_t * self.fs # 指定频率的移位序列
            #n0_i = array_distance / self.c * (self.f_unit * k )# shape: [M, 1]
            self.receive_signals_rfft[:,k:(k+1)] += signal_rfft[k] * np.exp(1j * 2 * np.pi * k * delta_n0 / self.L) / array_distance
            '''
            if k == 129:
                print(array_distance)
                print("k=129时")
                print("信号的相位角：", np.angle(signal_rfft[k]))
                print("第一个mic的相位角:", np.angle(self.receive_signals_rfft[0,k]))
                print("第二个mic的相位角:", np.angle(self.receive_signals_rfft[1,k]))
                print("相位移动：")
                print(np.angle(np.exp(1j * 2 * np.pi * k * delta_n0 / self.L)))
            '''
        
        # print("阵列接受到的频谱为[5, 513], :",self.receive_signals_rfft)

        # 更新阵列接受信号的协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1) 


    def set_signal(self, m, signal_rfft=None):
        self.receive_signals_rfft[m,:] = signal_rfft
        # 更新阵列接受信号的协方差矩阵
        self.covariance = np.dot(self.receive_signals_rfft, np.conj(self.receive_signals_rfft.T)) / (self.M - 1) 
    
    def beamform(self, x, y, z):
        distance_array = np.sqrt( (x - self.mic_x)**2  + (y - self.mic_y)**2 + (z - self.mic_z)**2 ) # 计算目标位置到各个阵元的距离， shape=[M, 1]
        t_delta = distance_array / self.c * self.fs
        # 遍历所有频点， 进行波束形成
        for k in range(self.receive_signals_rfft.shape[1]):
            if k < 1e-6: continue
            # 相位差: exp(-1j*2*np.pi*f*t), 其中FFT中, f = k * fs / self.L, t = distance_array / self.c
            phase_delta = np.exp(-1j * 2 * np.pi * k * t_delta / self.L) # shape: [M, 1]
            self.receive_signals_rfft[:,k:(k+1)] *= phase_delta

    def doa():
        pass

    def MVDR_polar_2D(self, rou_range, theta_range, z, step_rou=0.15, step_theta=5*180/np.pi):
        pass

    def MVDR_1D_x(self, x_range, y, z, step_x = 0.1):
        scan_x = np.arange(x_range[0], x_range[1], step_x)
        scan_y = y
        scan_z = z
        P_BF_1d = np.zeros((scan_x.shape[0],), dtype=complex)
        for i in range(0, scan_x.shape[0], 1):
            dis_i = np.sqrt( (scan_x[i]-self.mic_x)**2 + (scan_y-self.mic_y)**2 + (scan_z-self.mic_z)**2 ) # 计算扫描坐标到各个阵元的距离
            dis_ref = np.sqrt( (scan_x[i]-self.ref_x)**2 + (scan_y-self.ref_y)**2 + (scan_z-self.ref_z)**2 ) # 计算扫描坐标到参考阵元的距离
            dis_delta = dis_i - dis_ref # 计算扫描坐标到各个阵元和到参考阵元的程差矢量, shape: [M, 1]
            # 注意噢， 这里声源定位， 用的是讲相位差作用到信号上， 因此j前取正才能得到正确的位置。
            phase_delta = np.exp(1j * self.w * dis_delta/self.c) # 根据程差矢量计算相位差 shape: [M, 1]
            # 每个麦克风是独立测量的，这里认为协方差矩阵是可逆的，就不用伪逆的矩阵了。
            CovarianceInverse = np.linalg.inv(self.covariance)

            P_BF_1d[i] = 1 / np.abs(np.dot( phase_delta.T, np.dot(CovarianceInverse, phase_delta) ))
        return P_BF_1d
    
    def MVDR_1D_y(self, x, y_range, z, step_y = 0.1):
        scan_x = x
        scan_y = np.arange(y_range[0], y_range[1], step_y)
        scan_z = z
        P_BF_1d = np.zeros((scan_y.shape[0],), dtype=complex)
        for j in range(0, scan_y.shape[0], 1):
            dis_i = np.sqrt( (scan_x-self.mic_x)**2 + (scan_y[j]-self.mic_y)**2 + (scan_z-self.mic_z)**2 ) # 计算扫描坐标到各个阵元的距离
            dis_ref = np.sqrt( (scan_x-self.ref_x)**2 + (scan_y[j]-self.ref_y)**2 + (scan_z-self.ref_z)**2 ) # 计算扫描坐标到参考阵元的距离
            dis_delta = dis_i - dis_ref # 计算扫描坐标到各个阵元和到参考阵元的程差矢量, shape: [M, 1]
            # 注意噢， 这里声源定位， 用的是讲相位差作用到信号上， 因此j前取正才能得到正确的位置。
            phase_delta = np.exp(1j * self.w * dis_delta/self.c) # 根据程差矢量计算相位差 shape: [M, 1]
            #CovarianceInverse = np.linalg.inv(self.covariance)
            CovarianceInverse = np.linalg.pinv(self.covariance)
            P_BF_1d[j] = 1 / np.abs(np.dot( phase_delta.T, np.dot(CovarianceInverse, phase_delta)))
        return P_BF_1d # shape:[x_range.shape, ]

    def MVDR_2D(self, x_range, y_range, z, step_x = 0.1, step_y = 0.1):
        scan_x = np.arange(x_range[0], x_range[1], step_x)
        scan_y = np.arange(y_range[0], y_range[1], step_y)
        scan_z = z
        P_BF_2d = np.zeros((scan_x.shape[0], scan_y.shape[0]), dtype=complex)
        for i in range(0, scan_x.shape[0], 1):
            P_BF_2d[i,:] = self.MVDR_1D_y(scan_x[i], y_range, scan_z, step_y=step_y)
        
        return P_BF_2d # shape: [扫描x点数， 扫描y点数]

    '''
    def Music_2D(self, x_range, y_range, z, step_x=0.1, step_y = 0.1, num = None):
        # 该算法需要指定信号源数
        # 不指定信号源数， 则使用add_signals的数量
        if num == None:
            num = self.num_signals

        # 首先先奇异值分解
        #        
    '''     
    def beamforming_1D_x(self, x_range, y, z):
        scan_x = x_range
        scan_y = y
        scan_z = z
        P_BF_1d = np.zeros((scan_x.shape[0], self.L), dtype=complex) # shape = [N, num_freqs], 用来记录第i个位置, 第j个频点的值
        freq = np.arange(1, self.L) * self.fs / self.L # FFT对应的频率点
        for j in range(self.L): # 遍历每一个频点
            fk = freq[j]
            for i in range(0, scan_x.shape[0], 1):
                dis_i = np.sqrt( (scan_x[i]-self.mic_x)**2 + (scan_y-self.mic_y)**2 + (scan_z-self.mic_z)**2 ) # 计算扫描坐标到各个阵元的距离
                # 注意噢， 这里声源定位， 用的是讲相位差作用到信号上， 因此j前取负才能得到正确的位置。
                phase_delta = np.exp(-1j * 2 * np.pi * fk * dis_i/self.c) # 根据程差矢量计算相位差 shape: [M, 1]
                P_BF_1d[i][k] = np.abs(np.dot( phase_delta.T, np.dot(self.covariance, phase_delta)))
            
        return P_BF_1d # shape:[x_range.shape, ]

    def beamforming_1D_y(self, x, y_range, z):
        scan_x = x
        scan_y = y_range
        scan_z = z
        P_BF_1d = np.zeros((scan_y.shape[0], self.L), dtype=complex)
        freq = np.arange(1, self.L) * self.fs / self.L # FFT对应的频率点
        for j in range(self.L): # 遍历每一个频点
            fk = freq[j]
            for i in range(0, scan_y.shape[0], 1):
                dis_i = np.sqrt( (scan_x-self.mic_x)**2 + (scan_y[i]-self.mic_y)**2 + (scan_z-self.mic_z)**2 ) # 计算扫描坐标到各个阵元的距离
                # 注意噢， 这里声源定位， 用的是讲相位差作用到信号上， 因此j前取负才能得到正确的位置。
                phase_delta = np.exp(-1j * 2 * np.pi * fk * dis_i/self.c) # 根据程差矢量计算相位差 shape: [M, 1]
                P_BF_1d[i][k] = np.abs(np.dot( phase_delta.T, np.dot(self.covariance, phase_delta)))

        return P_BF_1d # shape:[x_range.shape, ]
            
        
    def beamforming_2D(self, x_range, y_range, z, step_x = 0.2, step_y = 0.2):
        scan_x = np.arange(x_range[0], x_range[1], step_x)
        scan_y = np.arange(y_range[0], y_range[1], step_y)
        scan_z = z
        P_BF_2d = np.zeros((scan_x.shape[0], scan_y.shape[0], self.L), dtype=complex)
        for i in range(0, scan_x.shape[0], 1):
            P_BF_2d[i,:] = self.beamforming_1D_y(scan_x[i], scan_y, scan_z)
        
        return P_BF_2d # shape: [扫描x点数， 扫描y点数]
    
    def beamforming_3D():
        pass

    def beamform_enhanced_cbf(self, x, y, z): # 该函数用波束形成方法实现对指定位置信号的增强。
        dis_i = np.sqrt( (x - self.mic_x)**2  + (y - self.mic_y)**2 + (z - self.mic_z)**2 ) # 计算目标位置到各个阵元的距离， shape=[M, 1]

        t_delta = dis_i / self.c # 根据程差矢量计算 每个阵元接收到信号的时间延迟， 用于后面计算相位偏移, shape: [M, 1]
        n0 = np.around(t_delta * self.fs, decimals = 0) # shape: [M, 1]
        # DFT的时移特性： x(n + n0) = X[k]*exp(j2*pi*k*n0/N)
        fft_receive_signals = np.fft.fft(self.receive_signals) # shape: [M, F]
        for k in range(fft_receive_signals.shape[1]):
            phase_delta = np.exp(-1j * 2 * np.pi * k * n0 / self.L) # shape: [M, 1]
            fft_receive_signals[:,k:(k+1)] *= phase_delta # 每一个频点乘对应的相位差

        enhanced_signals = np.fft.ifft(fft_receive_signals)
        enhanced_signals = np.sum(enhanced_signals, axis=0, keepdims=False) # shape [M, L] -> shape [L,]
        return NormAmplitude(np.abs(enhanced_signals))

    def beamform_enhanced_mvdr(self, x, y, z): # 该函数用MVDR方法实现对指定位置信号的增强
        pass

    def get_ref_audio(self, norm = True):
        if norm == True:
            if is_clipped(self.ref_signals[0,:]):
                return NormAmplitude(self.ref_signals[0, :])
        else:
            return self.ref_signals[0, :]

    def show_ref_audio(self, norm = True):
        pass

        
    def show_mic(self):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(self.mic_x, self.mic_y, self.mic_z, c = "r")
        #ax.set_xlim([300,700])
        #ax.set_ylim([300,700])
        #ax.set_zlim([0,self.z])
        plt.show()


if __name__ == "__main__":
    stage = 3
    if stage == 1: # show_mic
    # 初始化房间长宽高分别为10m, 10m, 4m
        x, y, z = 1000, 1000, 400 
        mic_x = [0, 3, 6, 9, 12, 15, 18, 21, 24, 12, 12, 12, 12, 12, 12, 12, 12]
        mic_y = [12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 3, 6, 9, 15, 18, 21, 24]
        mic_z = [0 for i in range(len(mic_x))]
        model = beamforming_model(mic_x, mic_y, mic_z, x, y, z)
        model.show_mic()
    if stage == 2: # add_signal, beamforming_2D
        from matplotlib import cm
        mic_x = [0, 3, 6, 9, 12, 15, 18, 21, 24, 12, 12, 12, 12, 12, 12, 12, 12]
        mic_y = [12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 3, 6, 9, 15, 18, 21, 24]
        mic_z = [0 for i in range(len(mic_x))]
        model = beamforming_model(mic_x, mic_y, mic_z)
        #model.show_mic()
        model.add_signal(12, 10, 10)
        P_BF = model.beamforming_2D([9,15], [9, 15], z=100, step_x=0.1, step_y=0.1)
        print(np.argmax(P_BF, axis=1))
        #print(P_BF)
        Pmax = np.max(P_BF)
        P_BF = P_BF / Pmax
        # 绘图
        x = np.arange(9,15,0.1)
        y = np.arange(9,15,0.1)
        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, P_BF, cmap=cm.coolwarm,
                            linewidth = 0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    if stage == 3: # 计算房间中， 信号源出于不同位置，所造成的阵列延时矩阵
        r = [0.000]
        r_ = [0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
        theta = [np.pi/4 * i for i in range(8)]

        mic_x = [0.000]
        mic_y = [0.000]
        for ri in r_:
            for thetai in theta:
                mic_x.append(ri * np.cos(thetai))
                mic_y.append(ri * np.sin(thetai))

        mic_z = [2.8 for i in range(len(mic_x))]
        model = beamforming_model(mic_x, mic_y, mic_z)
        h = 2.0 # 吸顶阵列麦高度为2.8m
        R = h * np.sqrt(3) # 在120度覆盖范围下， 我们的搜索圆半径范围。
        x = np.arange(-R, R, 0.1)
        y = np.arange(-R, R, 0.1)
        N = x.shape[0]
        delta_t_matrix = np.zeros((model.M, N))
        x = list(x)
        y = list(y)
        z = 0
        k = 0
        for i in x:
            for j in y:
                model.beamform_enhanced_cbf(i, k, z)

                
        
    