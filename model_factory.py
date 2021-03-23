"""
@author: Skye Cui
@file: model_factory.py
@time: 2021/3/23 13:50
@description: 
"""
import tensorflow as tf
from model import UTransformer


def model_factory(hp):
    x = tf.keras.Input(shape=(hp.in_seqlen, hp.height, hp.width, hp.num_predictor))
    y = tf.keras.Input(shape=(hp.out_seqlen, hp.height, hp.width, hp.num_predictor))
    ys = (y, y)

    utransformer = UTransformer(hp)
    outputs = utransformer([x, ys], training=True)

    model = tf.keras.Model(inputs=[x, ys], outputs=outputs)

    return model
