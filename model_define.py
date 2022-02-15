# modelDesign.py
#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_FEEDBACK_BITS = 512 #pytorch版本一定要有这个参数
CHANNEL_SHAPE_DIM1 = 126  # Nc
CHANNEL_SHAPE_DIM2 =  128 # Nt 发射天线数
CHANNEL_SHAPE_DIM3 = 2    # 实部虚部
BATCH_SIZE = 64    # 原始为128
FC_layer_size = 32256 # 126*128*2
#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    # 十进制数转为二进制数，位数为B
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        # 7 6 5 4 3 2 1 0
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        # integer的最后一个维度增加一维
        # 整数除以2的幂次取商，商模2取余数
        # 例如 整数5 5/1=5 5/2=2 5/4=1 5/8=0
        #           5%2=1 2%2=0 1%2=1 0%2=0
        # 倒序排列 0101 转为了二进制
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)  #  数的个数 x B
    return bit.type(torch.float32)
def Bit2Num(Bit, B):
    # B位二进制数转为十进制数
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    # 0101 转为 十进制
    # 1*1+0*2+1*4+0*8 = 0
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod  # 静态方法
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用，保存前向传播的变量。
    # 自己定义的Function中的forward()方法，
    # 所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，
    # autograd engine会自动将Variable unpack成Tensor，张量
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)  # 四舍五入 0-1转为0-16 再减0.5 相当于向下取值
        out = Num2Bit(out, B)              # 转为二进制
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    # 解量化
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)        # 转为数字 0-16
        out = (out + 0.5) / step   # 补偿0.5，归一化到0-1
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Encoder and Decoder Class Defining
# 编码和解码
def conv3x3(in_channels, out_channels, stride=1):
    # 定义3x3卷积，输入通道，输出通道，即有几个卷积核，步长为1
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    # 卷积核为3x3 步长为1 padding使卷积后维数相同 加偏置bias
class Encoder(nn.Module):
    num_quan_bits = 4 # 量化位数为4
    def __init__(self, feedback_bits): # 反馈位数为128
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(2, 2) # 定义输入为2通道输出为2通道的卷积
        self.conv2 = conv3x3(2, 2) # 定义输入为2通道输出为2通道的卷积
        self.conv3 = conv3x3(2, 2) # 定义输入为2通道输出为2通道的卷积
        self.fc = nn.Linear(FC_layer_size, int(feedback_bits / self.num_quan_bits)) # 126*128*2
        self.bn2_1 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        self.bn2_2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        # 定义全连接网络，1024转为128/4=32
        self.sig = nn.Sigmoid()
        # 定义激活函数
        self.quantize = QuantizationLayer(self.num_quan_bits)
        # 定义量化函数
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 维数换位，将通道数换到前面
        ########################################
        # 第一次
        x = self.conv1(x) # 卷积
        #x = self.bn2_1(x)
        #x = nn.BatchNorm2d(x).to(device=device)
        x = F.leaky_relu(x, negative_slope=0.3)
        #########################################
        # 第二次
        x = self.conv2(x)  # 卷积
        #x = self.bn2_2(x)
        #x = nn.BatchNorm2d(x).to(device=device)
        out = F.leaky_relu(x, negative_slope=0.3)
        #########################################
        x = self.conv2(x)  # 卷积
        out = F.leaky_relu(x, negative_slope=0.3)
        #########################################
        out = out.contiguous().view(-1, FC_layer_size) # 32256 = 126*128*2
        # 不改变数据维数，只是换一种索引方法，展成1024
        # 需使用contiguous().view(),或者可修改为reshape
        out = self.fc(out)  # 调用全连接网络转为32
        out = self.sig(out) # 调用SIGMOD函数激活
        out = self.quantize(out) # 量化
        return out
class Decoder(nn.Module):
    num_quan_bits = 4 # 量化位数为4
    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits # 定义反馈比特数为128
        self.dequantize = DequantizationLayer(self.num_quan_bits) # 定义解量化操作
        self.multiConvs = nn.ModuleList()  # 多次卷积函数
        self.fc = nn.Linear(int(feedback_bits / self.num_quan_bits), FC_layer_size) # TODO must change this 126*128*2
        # 定义从32到1024的全连接层
        self.out_cov = conv3x3(2, 2) # 定义输入2通道 输出2通道的卷积
        self.bn2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        self.sig = nn.Sigmoid() # 定义sigmod
        for _ in range(3):
            self.multiConvs.append(nn.Sequential(
                conv3x3(2, 8),
                #nn.BatchNorm2d(num_features=8, affine=True, track_running_stats=True), # should keep the same with the last layer
                nn.LeakyReLU(negative_slope=0.3),
                conv3x3(8, 16),
                #nn.BatchNorm2d(num_features=16, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.3),
                conv3x3(16, 16),
                nn.LeakyReLU(negative_slope=0.3),
                conv3x3(16, 2),
                #nn.BatchNorm2d(num_features=2, affine=True, track_running_stats=True)))
                ))

        # 进行三次卷积通道数 2->8->16->2
    def forward(self, x):
        out = self.dequantize(x) # 先解量化
        out = out.contiguous().view(-1, int(self.feedback_bits / self.num_quan_bits))
        # 转成32
        # 需使用contiguous().view(),或者可修改为reshape
        out = self.sig(self.fc(out))
        # 32转1024
        out = out.contiguous().view(-1, 2, CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2)  # TODO must change the number
        # 将通道提前到最前面
        # 需使用contiguous().view(),或者可修改为reshape
        #############################################
        # 第一次refine net
        residual = out
        for i in range(3):
            out = self.multiConvs[i](out)
        out = residual + out
        out = F.leaky_relu(out, negative_slope=0.3)
        # 第二次refine net
        residual = out
        for i in range(3):
            out = self.multiConvs[i](out)
        out = residual + out
        ################################################
        out = self.out_cov(out)
        #out = self.bn2(out)
        # 输入2 输出2 卷一次
        out = self.sig(out)
        # 归一化
        out = out.permute(0, 2, 3, 1)
        # 将通道换到最后一维
        return out
class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits) # 定义编码函数
        self.decoder = Decoder(feedback_bits) # 定义解码函数
    def forward(self, x):
        feature = self.encoder(x)    # 调用编码
        out = self.decoder(feature)  # 调用解码
        return out
#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining
def NMSE(x, x_hat):
    # 计算NMSE均方误差，x为实际值，x_hat为估计值
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    # 取x的实部层 len(x)是返回矩阵长度，不是元素个数
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    # 取x的虚部层
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    # 实部虚部同时减0.5是将(0,1)的数搬移至(-0.5,0.5)，使其中心为0
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    # axis= 1表示对第二外层[]里的最大单位块做块与块之间的运算,同时移除第二外层[]
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
def Score(NMSE):
    score = 1 - NMSE
    return score
#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData      # 矩阵类数据
    def __getitem__(self, index):
        return self.matdata[index]  # 索引得到元素
    def __len__(self):
        return self.matdata.shape[0]# 求len