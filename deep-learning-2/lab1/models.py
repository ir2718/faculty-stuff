import math
from typing import Union, Collection
from typing_extensions import TypeAlias

import tensorflow as tf

TFData: TypeAlias = Union[tf.Tensor, tf.Variable, float]

class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)
        neglog_pdf = 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)
        return neglog_pdf

    @tf.function
    def loss(self, data: TFData):
        nlog_normal = GMModel.neglog_normal_pdf(data, self.mean, self.logvar)
        nlog_categorical = tf.reduce_logsumexp(self.logpi) - self.logpi
        l = -tf.reduce_logsumexp(-(nlog_categorical + nlog_normal), axis=1)
        return l

    def p_xz(self, x: TFData, k: int) -> TFData:
        return tf.math.exp(-GMModel.neglog_normal_pdf(x, self.mean[k], self.logvar[k]))

    def p_x(self, x: TFData) -> TFData:
        w = tf.math.exp(self.logpi) / tf.reduce_sum(tf.math.exp(self.logpi))
        return tf.reduce_sum([w[k] * self.p_xz(x, k) for k in range(self.K)], axis=0)
