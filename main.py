"""
Dwa większe testy sieci Hopfielda z wieloma wzorcami.
Pierwszy operuje na losowych wzorcach i losowych danych.
  Generalnie w większości prób sieć osiąga wzorzec po 1, 2 max 3 iteracjach.
  Zdarzają się pojedyncze przypadki gdy sieć wcale nie osiąga stabilności po 100 iteracjach.
Drugi test operuje na danych MNIST z odręcznie pisanymi cyframi.
  Dane są importowane z biblioteki tensorflow.
  Sieć mimo działania na 784 neuronach wysypuje się już przy trzech wzorcach, nie jest w stanie rozpoznać nawet oryginalnych danych.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import hopfieldHeader
import random

### Testy z danymi losowymi ###
randPattern1 = np.random.choice([-1,1], size=144)
randPattern2 = np.random.choice([-1,1], size=144)
randPattern3 = np.random.choice([-1,1], size=144)
randPatterns = np.vstack((randPattern1, randPattern2, randPattern3))
randNet = hopfieldHeader.hopfieldNet(144)
randNet.learn(randPatterns)
randNet.plotWeights("randomWeights.png")
randNet.plotPatterns(12,"randomPattern")

randTestData1 = np.random.choice([-1,1], size=144)
plt.imshow(randTestData1.reshape(12,12))
plt.savefig("randTestData1.png")
randTestData2 = np.random.choice([-1,1], size=144)
plt.imshow(randTestData2.reshape(12,12))
plt.savefig("randTestData2.png")

result = randNet.test(randTestData1,100)
plt.imshow(result.reshape(12,12))
plt.savefig("randResult1.png")
result = randNet.test(randTestData2,100)
plt.imshow(result.reshape(12,12))
plt.savefig("randResult2.png")
###

### Import danych MNIST ###
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# wybór po jednym przykładzie z każdej cyfr
unique_labels = np.unique(train_labels)
selected_images = []
for label in unique_labels:
    label_indices = np.where(train_labels == label)[0]
    selected_image = train_images[label_indices[1]]
    selected_image = selected_image.reshape(-1)  #reshape macierzy na wektor
    #zmiana wartości w wektorze na -1 i 1
    selected_image = np.where(selected_image > 0, 1, -1)
    selected_images.append(selected_image)
###

### Testy z wzorcami MNIST ###
mnistPatterns1 = np.vstack((selected_images[3],selected_images[7]))
mnistPatterns2 = np.vstack((selected_images[0],selected_images[2],selected_images[4]))
mnistPatterns3 = np.vstack((selected_images[0],selected_images[7],selected_images[6],selected_images[2]))
mnistPatterns4 = selected_images

mnistNet1 = hopfieldHeader.hopfieldNet(784)
mnistNet1.learn(mnistPatterns1)
mnistNet1.plotWeights("mnistWeights.png")
mnistNet1.plotPatterns(28,"mnistPattern")
result = mnistNet1.test(selected_images[3],100)
plt.imshow(result.reshape(28, 28))
plt.savefig('mnistResult1.png')

mnistNet2 = hopfieldHeader.hopfieldNet(784)
mnistNet2.learn(mnistPatterns2)
mnistNet2.plotWeights("mnistWeights(2).png")
mnistNet2.plotPatterns(28,"mnistPattern(2)")
result = mnistNet2.test(selected_images[2],100)
plt.imshow(result.reshape(28, 28))
plt.savefig('mnistResult2.png')
#sieć wysypuje się przy trzech wzorcach, nie jest w stanie rozpoznać nawet czystych danych
###
