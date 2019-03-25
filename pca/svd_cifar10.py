# coding:utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# import glob
import numpy as np
from PIL import Image
# import cPickle
import pickle
# from sklearn.decomposition import PCA

# src = ROOT_DIR+'/../data/cifar-10-batches-py/test_batch'
src = ROOT_DIR+'/../data/cifar-10-batches-py/data_batch'
meta = ROOT_DIR+'/../data/cifar-10-batches-py/batches.meta'


Xraw, Y = None, None

for i in range(5):
  batch = "{}_{}".format(src, i+1)
  with open(batch, 'rb') as fo:
    raw_dict = pickle.load(fo, encoding='bytes')
    if Xraw is None:
      Xraw = raw_dict[b'data']
      Y = raw_dict[b'labels']
    else:
      Xraw = np.concatenate((Xraw, raw_dict[b'data']), axis=0)
      Y.extend(raw_dict[b'labels'])

### 这里的样本越多，奇异值分解后获得的奇异向量越好，降维d就可以选择地越小，但计算协方差矩阵量会很大
### 转换成浮点型np才会并行执行dot，整型不会
X = Xraw[:].astype(np.float32)
print("X.shape:{}".format(Xraw.shape))
print("X:{}".format(X))
# print(Y)


label_names = []
with open(meta, 'rb') as fo:
  label_names = pickle.load(fo, encoding='bytes')[b'label_names']

def trans_cifar10_to_img(x):
  height, width, channel = 32, 32, 3
  img = np.reshape(x, (channel,height,width)).transpose((1,2,0))
  return Image.fromarray(img, 'RGB')
  

# ### 按列中心化，每一列看做一维
mu = np.mean(Xraw, axis=0)
# print(mu)
X -= mu
print("Centered X:{}".format(X))

# sys.exit()

### 计算协方差矩阵
cov = np.dot(X.T, X)/X.shape[0]
# print(cov.shape)

### 奇异值分解
U, S, V = np.linalg.svd(cov)
# print(U.shape)
# print(V.shape)

### 降维
### 如果d取D=3*32*32=3072，则就没有降维，只是将图像转换到新的坐标系，然后再转换回来，图像理应不变，可以作为验证
# d = 3072
d = 144

### decorelate data 相当于坐标系变换，变换到求出来的特征向量作为基的坐标系上
# Xrot = np.dot(X, U)
# Xrot_reduced = Xrot[:,:d]

### 上面这两步合起来相当于
Xrot_reduced = np.dot(X, U[:,:d])

### 把降维后的数据再转换回原坐标系
Xret = np.dot(Xrot_reduced, U.transpose()[:d,:])

### 把恢复后的数据转成rgb类型并去中心化到0-255范围
Xret = (Xret+mu).astype(np.uint8)
# print(Xret)

if len(sys.argv)>1:
  check_idx = int(sys.argv[1])
  for i in range(check_idx):
      print("Y[{}]:{}".format(i, label_names[Y[i]]))
      trans_cifar10_to_img(Xraw[i]).show()
      # trans_cifar10_to_img(Xraw[i]).save('./result/{}_origin.jpg'.format(i))
      # trans_cifar10_to_img(Xret[i]).save('./result/{}_reduce.jpg'.format(i))


