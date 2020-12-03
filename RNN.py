# # Redes Neuronales Recurentes
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

numero_entradas = 2
numero_neuronas = 3

x0 = tf.placeholder(tf.float32,[None,numero_entradas])
x1 = tf.placeholder(tf.float32,[None,numero_entradas])

Wx = tf.Variable(tf.random_normal(shape=[numero_entradas,numero_neuronas]))
Wy = tf.Variable(tf.random_normal(shape=[numero_neuronas,numero_neuronas]))
b = tf.Variable(tf.zeros([1,numero_neuronas]))

y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)

lote_x0 = np.array([ [0,1],[2,3],[4,5] ])
lote_x1 = np.array([ [2,4],[3,9],[4,1] ])

init = tf.global_variables_initializer()

with tf.Session() as sesion:
    sesion.run(init)
    salida_y0,salida_y1 = sesion.run([y0,y1], feed_dict={x0:lote_x0,x1:lote_x1})

salida_y0
salida_y1