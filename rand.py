"""
Test sieci z 81 neuronami i 3 losowymi wzorcami.
Jako dane wejściowe wykorzystuję znikeształcone wzorce.
Sieć zdaje się prawie zawsze "osiągać" jakiś wzorzec,
przy wielokrotnych testach tylko raz nie udało się jej ustabilizować.
"""
import matplotlib.pyplot as plt
import numpy as np
import hopfieldHeader
import random

randPatterns = [
  np.random.choice([-1,1], size=81),
  np.random.choice([-1,1], size=81),
  np.random.choice([-1,1], size=81)
]

randNet = hopfieldHeader.hopfieldNet(81)
randNet.learn(randPatterns)
randNet.plotWeights("randWeights.png")
randNet.plotPatterns(9,"randPattern")

noisedData = [
  hopfieldHeader.noise(randPatterns[0], 5),
  hopfieldHeader.noise(randPatterns[1], 5),
  hopfieldHeader.noise(randPatterns[2], 5),
  hopfieldHeader.noise(randPatterns[0], 20),
  hopfieldHeader.noise(randPatterns[1], 20),
  hopfieldHeader.noise(randPatterns[2], 20),
  hopfieldHeader.noise(randPatterns[0], 35),
  hopfieldHeader.noise(randPatterns[1], 35),
  hopfieldHeader.noise(randPatterns[2], 35),
  hopfieldHeader.noise(randPatterns[0], 50),
  hopfieldHeader.noise(randPatterns[1], 50),
  hopfieldHeader.noise(randPatterns[2], 50)
]

for i in range(len(noisedData)):
  plt.imshow(noisedData[i].reshape(9, 9))
  plt.savefig(f'randNoised{i+1}.png')
  #plt.show()

  result = randNet.test(noisedData[i])
  plt.imshow(result.reshape(9, 9))
  plt.savefig(f'randResult{i+1}.png')
  #plt.show()
