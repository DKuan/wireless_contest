import numpy as np
from modelDesign import *
import torch
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = NUM_FEEDBACK_BITS #128
CHANNEL_SHAPE_DIM1 = 16
CHANNEL_SHAPE_DIM2 = 32
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
import h5py
mat = h5py.File('./channelData/Hdata.mat',"r")
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# reshape 到320000x32x16x2
H_test = data
test_dataset = DatasetFolder(H_test)
# 转成DatasetFolder类
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
# 转成方便训练的数据格式 一次编码512组数据 每次不打乱数据 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
# 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中
#=======================================================================================================================
#=======================================================================================================================
# Model Loading
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS).cuda()
model_encoder = autoencoderModel.encoder # 取编码的这个属性
model_encoder.load_state_dict(torch.load('./modelSubmit/encoder.pth.tar')['state_dict'])
# load_state_dict是net的一个方法
# 是将torch.load加载出来的数据加载到net中
# load
# 加载的是训练好的模型
print("weight loaded")
#=======================================================================================================================
#=======================================================================================================================
# Encoding
model_encoder.eval()
encode_feature = []
with torch.no_grad():
    for i, autoencoderInput in enumerate(test_loader):
        # 一次处理512组数据 一共处理625次 共计320000组数据
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = model_encoder(autoencoderInput)
        autoencoderOutput = autoencoderOutput.cpu().numpy()
        if i == 0:
            encode_feature = autoencoderOutput
        else:
            encode_feature = np.concatenate((encode_feature, autoencoderOutput), axis=0)
            # concatenate数组拼接 把每一次得到的结果合并起来 按列的方向合并
            
print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./encOutput.npy', encode_feature)
print('Finished!')