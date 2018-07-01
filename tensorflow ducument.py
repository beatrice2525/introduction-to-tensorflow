# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##定义计算
import tensorflow as tf
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')
result=a+b

#获取默认计算图以及查看一个运算所属的计算图
print(a.graph is tf.get_default_graph())
sess=tf.Session()
sess.run(result)




#通过tf.Graph函数生成新的计算图，在不同的计算图上定义和使用变量
g1=tf.Graph()
with g1.as_default():
    v=tf.get_variable('v',initializer=tf.zeros_initializer()(shape=[1]))

g2=tf.Graph()
with g2.as_default():
    v=tf.get_variable('v',initializer=tf.ones_initializer()(shape=[1]))
    
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))
        
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))
        
#使用GPU跑        
g=tf.Graph()
with g.device('/gpu:0'):
    result=a+b


#张量保存的是计算’张量‘的计算过程
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')
result=tf.add(a,b,name='add')
print(result)
#设置计算对象类型
c=tf.constant([1,2],name='c',dtype=tf.float32)
d=tf.constant([2.0,3.0],name='d')
result1=tf.add(c,d,name='add')
print(result1)
#设置回话并指定默认，通过tf.tensor.eval函数来计算一个张量的取值
with sess.as_default():
    print(result.eval())
#其他方法
print(sess.run(result))
#or
print(result.eval(session=sess))


#在python脚本 or jupyter下设置,需要关闭回话(我似乎操作不来)
sess=tf.InteractivateSession()
print(result.eval())
sess.close()

#通过 ConfigProte Protocal Buffer 来配置需要生成的对话
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
sess1=tf.InteractiveSession(config=config)
sess2=tf.Session(config=config)

        
#tensorflow官方文档附带DOCX的程序    
import numpy as np
x_data=np.float32(np.random.rand(2,10))
y_data=np.dot([0.100,0.200],x_data)+0.300
b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(w,x_data)+b


loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(0,201):
    sess.run(train)
    if step % 20==0:
        print (step, sess.run(w), sess.run(b))


















