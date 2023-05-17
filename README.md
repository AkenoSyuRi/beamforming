# beamforming
使用Python语言，支持任意空间阵列形状，任意声源的阵列信号处理仿真，内置宽带MVDR，宽带CBF，宽带MUSIC算法的实现。 并对音频进行了流式处理的逻辑实现。

## demo展示
效果展示见./example_audios/

其中bf_input.wav是空间中均匀圆阵mic接受信号后直接相加的结果(未经过任何处理), bf_output.wav是mic接受信号后进行beamforming的结果。 具体实现参考./audio_beamform_demo.ipynb。

作者为空间圆阵设计了俯仰角抑制算法，其中angle_filter_input.wav是未经过处理简单相加的结果， angle_filter_out.wav是经过处理之后的结果。 另外另一组示例，是长语音的demo。 具体实现参考./audio_denoise_demo.ipynb。

声源定位算法见audio_doa_demo.ipynb，其中Beamform/CMA100.py提供了CBF， MVDR， MUSIC三种宽带声源定位实现。

## Quick Start

这部分嘛，后续完善。。
