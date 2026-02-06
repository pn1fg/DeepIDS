# 模型架构说明

## 架构概览（文字描述）

该模型为多层全连接神经网络，用于41维特征的多分类任务。网络由若干线性层堆叠组成，每个隐层包含线性变换、批归一化、ReLU激活与Dropout正则化，输出层为线性层用于类别打分。

## 层级结构

- 输入层：41维特征向量
- 隐层1：Linear(41 → 256) + BatchNorm + ReLU + Dropout(0.3)
- 隐层2：Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
- 隐层3：Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
- 输出层：Linear(64 → 5)

## 推理路径

输入特征依次通过各隐层的线性变换与非线性激活，最终输出5维logits向量，可接Softmax得到各类别概率。
