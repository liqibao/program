# -*- coding:utf-8 -*-
'''
software: scikit_learn
Time    : 2018/10/14 22:24
Author  :{liqibao}
E-mail  :{931470099@qq.com}
'''
"""
Smote算法，解决类别不均衡问题
"""

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
class Smote:
    def __init__(self, samples, N = 10, k = 5):
        self.n_samples,self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0

    def over_sampling(self):
        N = int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors = self.k).fit(self.samples)
        print("neighbors",neighbors)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1,-1), return_distance = False)[0]
            self._populate(self, N, i, nnarray)
        return self.synthetic

    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samples[nnarray[nn] - self.samples[i]]  # 计算随机点和原始点的差值
            gap = random.random()           # 生成[0,1]之间的随机数
            self.synthetic[self.newindex] = self.samples[i] + gap * dif   # 差值乘以随机数加上原始点得到新的样本点
            self.newindex += 1