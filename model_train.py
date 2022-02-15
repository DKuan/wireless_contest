'''
Author: your name
Date: 2021-12-22 23:12:28
LastEditTime: 2021-12-26 17:23:48
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \csdn_code\model_train.py
'''
import os
import time
import numpy as np
import torch
import fitlog
from model_define import AutoEncoder,DatasetFolder #*
import torch.nn as nn
import scipy.io as sio  # 无法导入MATLABv7版本的mat
############hyper
record_info = "加大网络层数，多加了一层"
NUM_FEEDBACK_BITS = 512   # 反馈数据比特数
CHANNEL_SHAPE_DIM1 = 126   # Nc
CHANNEL_SHAPE_DIM2 = 128   # Nt 发射天线数
CHANNEL_SHAPE_DIM3 = 2    # 实部虚部
BATCH_SIZE = 64    # 原始为128
EPOCHS = 1400
LEARNING_RATE = 1e-3
PRINT_RREQ = 100    # 每100个数据输出一次
############hyper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fitlog.commit(__file__)             # auto commit your codes
fitlog.set_log_dir('logs/')         # set the logging directory
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters

torch.manual_seed(1) # 随机种子初始化神经网络
# Data Loading
mat = sio.loadmat('.\\data\\Htrain.mat')
data = mat['H_train']
import h5py

#mat = h5py.File("D:\\AI_contest\\wireless_rebuild\\csdn_code\\channel_data\\Htrain.mat", "r") # 读数据
#data = np.transpose(mat)  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# data 320000x32x16x2
# data = np.transpose(data, (0, 3, 1, 2))
split = int(data.shape[0] * 0.7)
# 70% 数据训练，30% 数据测试
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
# 将data_train转为DatasetFolder类，可以调用其中的方法
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
# 320000*0.7/16 = 14000
# 每个batch加载多少个数据 设置为True时会在每个epoch重新打乱数据 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
# 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS).to(device)   # 调用自动编码解码函数
autoencoderModel = autoencoderModel.cuda().to(device)       # 转为GPU
criterion = nn.MSELoss().cuda().to(device)             # 用MSE作为损失函数 转为GPU
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)
# 优化器 ：Adam 优化算法 # 调用网络参数
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1  # 最大损失，小于最大损失才会保存模型
for epoch in range(EPOCHS):
    autoencoderModel.train().to(device)
    # 开始训练 启用 Batch Normalization 和 Dropout。
    # Dropout是随机取一部分网络连接来训练更新参数
    for i, autoencoderInput in enumerate(train_loader):
        # 遍历训练数据
        autoencoderInput = autoencoderInput.cuda().to(device)            # 转为GPU格式
        autoencoderOutput = autoencoderModel(autoencoderInput).to(device) # 使用模型求输出
        loss = criterion(autoencoderOutput, autoencoderInput).to(device)  # 输入输出传入评价函数 求均方误差
        optimizer.zero_grad()  # 清空过往梯度；
        loss.backward()        # 反向传播，计算当前梯度；
        optimizer.step()       # 根据梯度更新网络参数
        fitlog.add_loss(loss.item(),name="Loss",step=i+epoch*BATCH_SIZE)
        if i % PRINT_RREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.7f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
    # Model Evaluating 模型评估
    autoencoderModel.eval().to(device)
    # 不启用 Batch Normalization 和 Dropout
    # 是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
    totalLoss = 0
    # torch.no_grad()内的内容，不被track梯度
    # 该计算不会在反向传播中被记录。
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            # 加载测试数据
            autoencoderInput = autoencoderInput.cuda().to(device)             # 转为GPU格式
            autoencoderOutput = autoencoderModel(autoencoderInput).to(device) # 求输出
            totalLoss += criterion(autoencoderOutput, autoencoderInput).item() * autoencoderInput.size(0)
            # size（0）就是batch size的大小
            # 用测试数据来测试模型的损失

        averageLoss = totalLoss / len(test_dataset) # 求每一个EPOCHS后的平均损失
        fitlog.add_loss(averageLoss,name="eval_loss",step=epoch*BATCH_SIZE)

        if averageLoss < bestLoss:   # 平均损失如果小于1才会保存模型
            # Model saving
            time_now = time.strftime("%d%H%M%S")
            # Encoder Saving
            torch.save({'state_dict': autoencoderModel.encoder.state_dict(), }, 
                       '.\\Modelsave\\new_train\\encoder_1225{}.pth.tar'.format(time_now)
                       )
            # Decoder Saving
            torch.save({'state_dict': autoencoderModel.decoder.state_dict(), }, 
                       '.\\Modelsave\\new_train\\decoder_1225{}.pth.tar'.format(time_now)
                       )
            print("Model saved, the averageLoss is", averageLoss, time_now)
            bestLoss = averageLoss   # 更新最大损失，使损失小于该值是才保存模型
# finish the fitlog
fitlog.finish() 