# beamforming
使用Python，支持任意空间阵列形状，任意声源位置的阵列信号处理仿真，内置宽带MVDR，宽带CBF，宽带MUSIC算法的实现。 并对音频进行了流式处理的逻辑实现。

## demo展示
效果展示见./example_audios/

其中bf_input.wav是空间中非均匀圆阵mic接受信号后直接相加的结果(未经过任何处理), bf_output.wav是mic接受信号后进行beamforming的结果。 具体实现参考./audio_beamform_demo.ipynb。

作者为空间圆阵设计了俯仰角抑制算法，其中angle_filter_input.wav是未经过处理简单相加的结果， angle_filter_out.wav是经过处理之后的结果。 另外另一组示例，是长语音的demo。 具体实现参考./audio_denoise_demo.ipynb。
单纯进行俯仰角抑制后多mic信号相加，因为每个mic接受的信号相位有一些差异，会导致类似混响的听感，此时进行beamforming可以消混响，同时进一步加强特定方位的信号，实现进一步的降噪。参考angle_filter+bf_output.wav。

为什么单纯进行俯仰角抑制会有一些残余呢？事实上，如果角度内的信号和角度外的信号能量分布在不同时间，那么我目前的算法能够完全消掉俯仰角之外的信号(见长语音demo)。但如果他们在某一个时刻在相同频点同时有能量，此时他们的相位会混叠成一个新的相位，难以判断(具体见三角函数和差化积公式，曾经小小推导了一下，不是个线性方程组)，这一部分是比较令我头疼的一个点。

声源定位算法见audio_doa_demo.ipynb，其中Beamform/CMA100.py提供了CBF， MVDR， MUSIC三种宽带声源定位实现。

具体部署C++后续可能会更新。

## Quick Start

这部分后续作者有空完善。
