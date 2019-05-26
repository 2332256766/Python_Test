import numpy as np

def debugInitializeWeights(fan_out, fan_in): # 扇入 扇出
    ''''''
    W = np.zeros((fan_out, 1+fan_in)) # x行y列
    return np.sin(np.arange(W.size)+1).reshape(W.shape)/10
