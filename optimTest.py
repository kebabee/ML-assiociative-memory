"""
Wrzutka jako backup
"""

import matplotlib.pyplot as plt
import numpy as np
import hopfieldHeader
from PIL import Image

A = np.where(np.array(Image.open('litery/A.png'))[:,:,0]<128,1,-1).reshape(-1)
C = np.where(np.array(Image.open('litery/C.png'))[:,:,0]<128,1,-1).reshape(-1)
O = np.where(np.array(Image.open('litery/O.png'))[:,:,0]<128,1,-1).reshape(-1)
R = np.where(np.array(Image.open('litery/R.png'))[:,:,0]<128,1,-1).reshape(-1)
T = np.where(np.array(Image.open('litery/T.png'))[:,:,0]<128,1,-1).reshape(-1)
U = np.where(np.array(Image.open('litery/U.png'))[:,:,0]<128,1,-1).reshape(-1)
X = np.where(np.array(Image.open('litery/X.png'))[:,:,0]<128,1,-1).reshape(-1)
Z = np.where(np.array(Image.open('litery/Z.png'))[:,:,0]<128,1,-1).reshape(-1)

lettersPatterns = [A,C,O,R,T,U,X,Z]

network1 = hopfieldHeader.hopfieldNet(256)
network1.addPatterns([R,O,X,Z,T])
# network.optimLearn2([A,X,O,Z,U,R,T], 1)


noisedData = [
  hopfieldHeader.noise(A, 10),
  hopfieldHeader.noise(X, 10),
  hopfieldHeader.noise(O, 10),
  hopfieldHeader.noise(Z, 10),
  hopfieldHeader.noise(U, 10),
  hopfieldHeader.noise(R, 10),
  hopfieldHeader.noise(T, 10),
  hopfieldHeader.noise(A, 25),
  hopfieldHeader.noise(X, 25),
  hopfieldHeader.noise(O, 25),
  hopfieldHeader.noise(Z, 25),
  hopfieldHeader.noise(U, 25),
  hopfieldHeader.noise(R, 25),
  hopfieldHeader.noise(T, 25),
  hopfieldHeader.noise(A, 25),
  hopfieldHeader.noise(X, 40),
  hopfieldHeader.noise(O, 40),
  hopfieldHeader.noise(Z, 40),
  hopfieldHeader.noise(U, 40),
  hopfieldHeader.noise(R, 40),
  hopfieldHeader.noise(T, 40)
  # hopfieldHeader.noise(R, 1)
]

noisedData1 = [
  # hopfieldHeader.noise(A, 10),
  hopfieldHeader.noise(X, 10),
  hopfieldHeader.noise(O, 10),
  hopfieldHeader.noise(Z, 10),
  # hopfieldHeader.noise(U, 10),
  hopfieldHeader.noise(R, 10),
  hopfieldHeader.noise(T, 10),
  # hopfieldHeader.noise(A, 25),
  hopfieldHeader.noise(X, 25),
  hopfieldHeader.noise(O, 25),
  hopfieldHeader.noise(Z, 25),
  # hopfieldHeader.noise(U, 25),
  hopfieldHeader.noise(R, 25),
  hopfieldHeader.noise(T, 25),
  # hopfieldHeader.noise(A, 25),
  hopfieldHeader.noise(X, 40),
  hopfieldHeader.noise(O, 40),
  hopfieldHeader.noise(Z, 40),
  # hopfieldHeader.noise(U, 40),
  hopfieldHeader.noise(R, 40),
  hopfieldHeader.noise(T, 40)
]

noisedData2 = [
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
  np.random.choice([-1,1], size=256),
]

for i in range(10):
  network1.optimLearn6(R, 10)
# network1.plotWeights("d")

for i in range(10):
  network1.optimLearn6(O, 3)
# network1.plotWeights("d")

for i in range(10):
  network1.optimLearn6(X, 4)
# network1.plotWeights("d")

for i in range(10):
  network1.optimLearn6(Z, 1)
# network1.plotWeights("d")

for i in range(10):
  network1.optimLearn6(T, 0.3)
network1.plotWeights("wagiOptim")
network1.plotPatterns(16, "pattern")

"""
# result = network.asynTest(noisedData1[7])
# plt.imshow(result.reshape(16, 16), cmap='gray')
#   # plt.savefig(f'wyniki/lettersResultSynch{i+1}.png')
# plt.show()

# result = network.asynTest(noisedData1[8])
# plt.imshow(result.reshape(16, 16), cmap='gray')
#   # plt.savefig(f'wyniki/lettersResultSynch{i+1}.png')
# plt.show()

# result = network.asynTest(noisedData1[9])
# plt.imshow(result.reshape(16, 16), cmap='gray')
#   # plt.savefig(f'wyniki/lettersResultSynch{i+1}.png')
# plt.show()

# result = network.asynTest(noisedData1[10])
# plt.imshow(result.reshape(16, 16), cmap='gray')
#   # plt.savefig(f'wyniki/lettersResultSynch{i+1}.png')
# plt.show()

# 
# network.optimLearn4([A,X,O,Z,U,R,T], 0.1)
# network.optimLearn2(noisedData1, 0.01)




# network.optimLearn1(noisedData, 0.1)
# network.optimLearn1(noisedData2, 0.01)

# network.plotPatterns(16,"pattern")
# network.plotWeights("z")
"""

network2 = hopfieldHeader.hopfieldNet(256)
network2.learn([R,O,X,Z,T])
network2.plotWeights("wagiStandard")
# print(network2.weights)

for i in range(len(noisedData1)):
  plt.imshow(noisedData1[i].reshape(16, 16), cmap='gray')
  # plt.savefig(f'wyniki1/lettersNoised{i+1}.png')
  plt.show()

  result = network1.asynTest(noisedData1[i])
  plt.imshow(result.reshape(16, 16), cmap='gray')
  # plt.savefig(f'wyniki1/lettersResultAsynchOptim{i+1}.png')
  plt.show()

for i in range(len(noisedData1)):
  plt.imshow(noisedData1[i].reshape(16, 16), cmap='gray')
  # plt.savefig(f'wyniki/lettersNoised{i+1}.png')
  plt.show()

  result = network2.asynTest(noisedData1[i])
  plt.imshow(result.reshape(16, 16), cmap='gray')
  # plt.savefig(f'wyniki1/lettersResultAsynch{i+1}.png')
  plt.show()
