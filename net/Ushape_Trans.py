# -*- coding: utf-8 -*-
# @File    : Ushape_Trans.py
# coding=utf-8
# Design based on the pix2pix

import torch.nn as nn
import torch.nn.functional as F 
import torch
import datetime
import os
import time
import timeit
import copy
import numpy as np 
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from net.block import *
from net.block import _equalized_conv2d
from net.SGFMT import TransformerModel
from net.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from net.CMSFFT import ChannelTransformer
from net.Agent_Attention import AgentAttention
from net.ViPT import PatchEmbed
from torch.nn import MultiheadAttention
# 增加梯度裁剪和混合精度训练支持
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()  # 用于混合精度训练

##权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.grl = GradientReversal(alpha=1.0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.grl(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# class StandardAttention(nn.Module):
# 	def __init__(self, dim=512, num_heads=8):
# 		super().__init__()
# 		self.norm = nn.LayerNorm(dim)
# 		self.attention = MultiheadAttention(
# 			embed_dim=dim,
# 			num_heads=num_heads,
# 			dropout=0.1,
# 			batch_first=True
# 		)
#
# 	def forward(self, x):
# 		"""
#         x: [B, N, C] 形状的输入token
#         返回: [B, N, C] 形状的注意力输出
#         """
# 		x = self.norm(x)
# 		attn_output, _ = self.attention(x, x, x)
# 		return attn_output

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		# 输入: 1, 3, 256, 256
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 1, 64, 128, 128
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 1, 128, 64, 64
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 1, 256, 32, 32
		self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 1, 512, 16, 16

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		x = self.conv4(x)
		return x


class Generator(nn.Module):
	"""
	MSG-Unet-GAN的生成器部分
	"""
	def __init__(self,
		img_dim=256,
		patch_dim=16,
		embedding_dim=512,
		num_channels=3,
		num_heads=8,
		num_layers=4, #
		hidden_dim=256,
		dropout_rate=0.0,
		attn_dropout_rate=0.0,
		in_ch=3, 
		out_ch=3,

		conv_patch_representation=True,
		positional_encoding_type="learned", # 'fixed',
		use_eql=True):
		super(Generator, self).__init__()
		assert embedding_dim % num_heads == 0
		assert img_dim % patch_dim == 0

		self.out_ch=out_ch #输出通道数
		self.in_ch=in_ch #输入通道数
		self.img_dim = img_dim   #输入图片尺寸
		self.embedding_dim = embedding_dim  #512
		self.num_heads = num_heads  #多头注意力中头的数量
		self.patch_dim = patch_dim  #每个patch的尺寸
		self.num_channels = num_channels  #图片通道数?
		self.dropout_rate = dropout_rate  #drop-out比率
		self.attn_dropout_rate = attn_dropout_rate  #注意力模块的dropout比率
		self.conv_patch_representation = conv_patch_representation  #True

		self.num_patches = int((img_dim // patch_dim) ** 2)  #将三通道图片分成多少块
		self.seq_length = self.num_patches  #每个sequence的长度为patches的大小
		self.flatten_dim = 128 * num_channels  #128*3=384
		"""
		加一些自己的操作
		"""
		# self.six_to_three = nn.Sequential(
		# 	nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0),
		# 	nn.BatchNorm2d(3),
		# 	nn.ReLU(inplace=True)
		# )
		self.dropout_rate = 0.3
		self.dropout = nn.Dropout(p=self.dropout_rate)
		# self.patch_embed=PatchEmbed(in_chans=6)
		self.patch_embed1 = PatchEmbed()
		self.patch_embed2 = PatchEmbed()

		# self.patch_embed_p= PatchEmbed(in_chans=6,embed_dim=480,patch_size=32)
		dim = 512  # token 的维度
		window_size = (16, 16)  # 窗口大小
		num_heads = 8  # 注意力头数
		agent_num = 49  # 假设有 16 个代理

		# 创建 AgentAttention 实例
		self.attention_layer1 = AgentAttention(
			dim=dim,
			window_size=window_size,
			num_heads=num_heads,
			agent_num=agent_num
		)
		self.attention_layer2 = AgentAttention(
			dim=dim,
			window_size=window_size,
			num_heads=num_heads,
			agent_num=agent_num
		)
		# # 替换原来的AgentAttention初始化
		# self.attention_layer1 = StandardAttention(dim=512, num_heads=num_heads)
		# self.attention_layer2 = StandardAttention(dim=512, num_heads=num_heads)
		#线性编码
		self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
		#位置编码
		if positional_encoding_type == "learned":
			self.position_encoding = LearnedPositionalEncoding(
				self.seq_length, self.embedding_dim, self.seq_length
			)
		elif positional_encoding_type == "fixed":
			self.position_encoding = FixedPositionalEncoding(
				self.embedding_dim,
			)

		self.pe_dropout = nn.Dropout(p=self.dropout_rate)

		self.transformer = TransformerModel(
			embedding_dim, #512
			num_layers, #4
			num_heads,  #8
			hidden_dim,  #4096

			self.dropout_rate,
			self.attn_dropout_rate,
        )

		#layer Norm
		self.pre_head_ln = nn.LayerNorm(embedding_dim)

		if self.conv_patch_representation:

			self.Conv_x = nn.Conv2d(
				256,
				self.embedding_dim,  #512
				kernel_size=3,
				stride=1,
				padding=1
		    )

		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)



		#modulelist
		self.rgb_to_feature=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
		self.feature_to_rgb=ModuleList([to_rgb(32),to_rgb(64),to_rgb(128),to_rgb(256)])

		self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Conv1=conv_block(self.in_ch, 16)
		self.Conv1_1 = conv_block(16, 32)
		self.Conv2 = conv_block(32, 32)
		self.Conv2_1 = conv_block(32, 64)
		self.Conv3 = conv_block(64,64)
		self.Conv3_1 = conv_block(64,128)
		self.Conv4 = conv_block(128,128)
		self.Conv4_1 = conv_block(128,256)

		self.Conv5 = conv_block(512,256)

		#self.Conv_x = conv_block(256,512)
		self.mtc = ChannelTransformer(channel_num=[32,64,128,256],
									patchSize=[32, 16, 8, 4])
								

		self.Up5 = up_conv(256, 256)
		self.coatt5 = CCA(F_g=256, F_x=256)
		self.Up_conv5 = conv_block(512, 256)
		self.Up_conv5_1 = conv_block(256, 256)

		self.Up4 = up_conv(256, 128)
		self.coatt4 = CCA(F_g=128, F_x=128)
		self.Up_conv4 = conv_block(256, 128)
		self.Up_conv4_1 = conv_block(128, 128)

		self.Up3 = up_conv(128, 64)
		self.coatt3 = CCA(F_g=64, F_x=64)
		self.Up_conv3 = conv_block(128, 64)
		self.Up_conv3_1 = conv_block(64, 64)

		self.Up2 = up_conv(64, 32)
		self.coatt2 = CCA(F_g=32, F_x=32)
		self.Up_conv2 = conv_block(64, 32)
		self.Up_conv2_1 = conv_block(32, 32)

		self.Conv = nn.Conv2d(32, self.out_ch, kernel_size=1, stride=1, padding=0)

	# 新增域分类器
		self.domain_classifier = DomainClassifier(in_channels=256)  # 假设编码器输出256通道
		# self.polar_feature=ConvNet()  #   消融实验

	def reshape_output(self,x): #将transformer的输出resize为原来的特征图尺寸
		x = x.view(
			x.size(0),
			int(self.img_dim / self.patch_dim),
			int(self.img_dim / self.patch_dim),
			self.embedding_dim,
			)#B,16,16,512
		x = x.permute(0, 3, 1, 2).contiguous()

		return x

	def forward(self, x,p): #,device
		#print(x.shape)

		output=[]
		x_p=p
		# x=x_p
		# x=self.six_to_three(x)
		AOP_separated = x_p[:, :3, :, :]
		DOP_separated = x_p[:, 3:, :, :]

		x_1=self.Maxpool(x)   # 1，3，128，128
		x_2=self.Maxpool(x_1)  # 1，3，64，64
		x_3=self.Maxpool(x_2)  # 1，3，32，32

		e1 = self.Conv1(x)   # 1，16，256，256
		e1 = self.Conv1_1(e1) # 1，32，256，256
		e2 = self.Maxpool1(e1)

		x_1=self.rgb_to_feature[0](x_1)

		e2=x_1+e2
		e2 = self.Conv2(e2)
		e2 = self.Conv2_1(e2)
		e3 = self.Maxpool2(e2)
		#64*64*64

		x_2=self.rgb_to_feature[1](x_2)
		#e3=torch.cat((x_2,e3), dim=1)
		e3=x_2+e3
		e3 = self.Conv3(e3)
		e3 = self.Conv3_1(e3)
		e4 = self.Maxpool3(e3)
		#128*32*32

		x_3=self.rgb_to_feature[2](x_3)
		e4=x_3+e4
		e4 = self.Conv4(e4)
		e4 = self.Conv4_1(e4)
		e5 = self.Maxpool4(e4)
		#256*16*16

		#channel-wise transformer-based attention
		# 编码P
		# P_token = self.patch_embed_p(x_p)
		e1,e2,e3,e4,att_weights = self.mtc(e1,e2,e3,e4)  #,P_token

		#spatial-wise transformer-based attention
		residual=e5
		#中间的隐变量
		#conv_x应该接受256通道，输出512通道的中间隐变量
		e5= self.bn(e5)
		e5=self.relu(e5)
		e5= self.Conv_x(e5) #out->512*16*16 shape->B,512,16,16
		e5= e5.permute(0, 2, 3, 1).contiguous()  # B,512,16,16->B,16,16,512
		e5= e5.view(e5.size(0), -1, self.embedding_dim) #B,16,16,512->B,16*16,512 线性映射层
		e5= self.position_encoding(e5) #位置编码
		e5= self.pe_dropout(e5)	 #预dropout层
		# apply transformer

		# 转换辅助模态 到 token

		AOP_token=self.patch_embed1(AOP_separated) # 1,3,256,256->1,256,512
		DOP_token=self.patch_embed2(DOP_separated)

		# aop和dop进行agent attention操作
		aop=self.attention_layer1(AOP_token)
		dop=self.attention_layer2(DOP_token)

		# aop=self.polar_feature(AOP_separated)   # 消融实验的 no GPFE
		# dop=self.polar_feature(DOP_separated)   # 消融实验的

		# x_p=self.patch_embed(x_p)
		# polar=self.patch_embed(x_p)
		e5= self.transformer(e5,aop,dop ) # #,x_p # 输入e5维度 1,256,512   #   这个参数是Utrans8的,device
		e5= self.pre_head_ln(e5)
		e5 = self.dropout(e5)  # Apply dropout here after transformer output
		e5= self.reshape_output(e5)#out->512*16*16 shape->B,512,16,16
		e5=self.Conv5(e5) #out->256,16,16 shape->B,256,16,16
		#residual是否要加bn和relu？
		e5=e5+residual

		# 在编码器后获取特征
		encoder_features = e5  # 假设e5是编码器最终输出 [B,256,16,16]

		# 域分类预测
		domain_pred = self.domain_classifier(encoder_features)



		d5 = self.Up5(e5)
		# d5 = self.dark_attn5(d5)  # 新增
		e4_att = self.coatt5(g=d5, x=e4)
		d5 = torch.cat((e4_att, d5), dim=1)
		d5 = self.Up_conv5(d5)
		d5 = self.Up_conv5_1(d5)
		#256
		out3=self.feature_to_rgb[3](d5)
		output.append(out3)#32*32orH/8,W/8

		d4 = self.Up4(d5)
		# d4 = self.dark_attn4(d4)  # 新增
		e3_att = self.coatt4(g=d4, x=e3)
		d4 = torch.cat((e3_att, d4), dim=1)
		d4 = self.Up_conv4(d4)
		d4 = self.Up_conv4_1(d4)
		#128
		out2=self.feature_to_rgb[2](d4)
		output.append(out2)#64*64orH/4,W/4

		d3 = self.Up3(d4)
		# d3 = self.dark_attn3(d3)  # 新增
		e2_att = self.coatt3(g=d3, x=e2)
		d3 = torch.cat((e2_att, d3), dim=1)
		d3 = self.Up_conv3(d3)
		d3 = self.Up_conv3_1(d3)
		#64
		out1=self.feature_to_rgb[1](d3)
		output.append(out1)#128#128orH/2,W/2

		d2 = self.Up2(d3)
		# d2 = self.dark_attn2(d2)  # 新增
		e1_att = self.coatt2(g=d2, x=e1)
		d2 = torch.cat((e1_att, d2), dim=1)
		d2 = self.Up_conv2(d2)
		d2 = self.Up_conv2_1(d2)
		#32
		out0=self.feature_to_rgb[0](d2)
		output.append(out0)#256*256

		#out = self.Conv(d2)

		#d1 = self.active(out)
		#output=np.array(output)
		return output,domain_pred




class Discriminator(nn.Module):
    def __init__(self, in_channels=3,use_eql=True):
        super(Discriminator, self).__init__()

        self.use_eql=use_eql
        self.in_channels=in_channels


        #modulelist
        self.rgb_to_feature1=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
        self.rgb_to_feature2=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])


        self.layer=_equalized_conv2d(self.in_channels*2, 64, (1, 1), bias=True)
        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)


        self.layer0=DisGeneralConvBlock(64,64,use_eql=self.use_eql)
        #128*128*32
        
        self.layer1=DisGeneralConvBlock(128,128,use_eql=self.use_eql)
        #64*64*64
        
        self.layer2=DisGeneralConvBlock(256,256,use_eql=self.use_eql)
        #32*32*128
        
        self.layer3=DisGeneralConvBlock(512,512,use_eql=self.use_eql)
        #16*16*256
        
        self.layer4=DisFinalBlock(512,use_eql=self.use_eql)
        #8*8*512
        


    def forward(self, img_A, inputs):
    	#inputs图片尺寸从小到大
        # Concatenate image and condition image by channels to produce input
        #img_input = torch.cat((img_A, img_B), 1)
        #img_A_128= F.interpolate(img_A, size=[128, 128])
        #img_A_64= F.interpolate(img_A, size=[64, 64])
        #img_A_32= F.interpolate(img_A, size=[32, 32])


        x=torch.cat((img_A[3], inputs[3]), 1)
        y = self.pixNorm(self.lrelu(self.layer(x)))
        
        y=self.layer0(y)
        #128*128*64
        

        x1=self.rgb_to_feature1[0](img_A[2])
        x2=self.rgb_to_feature2[0](inputs[2])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer1(y)
        #64*64*128
        

        x1=self.rgb_to_feature1[1](img_A[1])
        x2=self.rgb_to_feature2[1](inputs[1])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer2(y)
        #32*32*256
        
        x1=self.rgb_to_feature1[2](img_A[0])
        x2=self.rgb_to_feature2[2](inputs[0])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer3(y)
        #16*16*512
        
        y=self.layer4(y)
        #8*8*512

        return y


