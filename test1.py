from rbm import BernoulliRBM
import numpy as np
from random import shuffle

matrices = []

for i in range(0,1024):
    matrices.append([1,0])

for i in range(0,128):
    matrices.append([0,1])

shuffle(matrices)

batch=np.array(matrices)

model = BernoulliRBM(n_components=256,verbose=1,n_iter=16,batch_size=16)
model.fit(batch)