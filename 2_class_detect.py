# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 20:27:07 2017

@author: Silence
"""
#导入TensorFlow库
import tensorflow as tf
#导入Numpy科学计算工具包
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义输入输出，数据量较小，可以将数据全部放入batch
#但当数据量较大时，可能产生内存溢出
#维度使用“None”可以方便使用不大的batch大小
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-output")

#定义神经网络前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数(交叉熵)和反向传播算法
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成模拟训练集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)]for (x1,x2) in X]

#创建会话运行tf程序
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  print(sess.run(w1))
  print(sess.run(w2))
  
  #设定训练轮次
  STEPS = 5000
  for i in range(STEPS):
    #每次选取batch_size个样本进行训练
    start = (i*batch_size) % dataset_size
    end = min(start+batch_size,dataset_size)
    
    #通过选取的样本训练神经网络并更新参数
    sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
    if i%1000 == 0:
      #每隔一段时间计算在所有数据上的交叉熵并输出
      total_cross_entropy = sess.run(cross_entropy,feed_dict = {x:X,y_:Y})
      print("After %d training steps,cross entropy on all data is %g"%(i,total_cross_entropy))
  print(sess.run(w1))
  print(sess.run(w2))
      
      
      
