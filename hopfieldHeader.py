"""
Plik nagłówkowy implementujący klasę hopfieldNet.
Klasa zawiera konstruktor tworzący obiekt zawierający liczbę neuronów, macierz wag i macierz z wyuczonymi wzorcami.
Metoda learn implementuje uczenie sieci z wykorzystaniem iloczynu macierzy dla każdego przyjętego wzorca.
Metoda test implementuje działanie sieci zgodnie z modyfikacją synchroniczną, tzn:
  1. wektor NewData początkowo przechowuje aktywacje wszystkich neuronów obliczone za pomoca iloczynu macierzy
  2. każdy element wektora jest przekszałcany za pomocą funkcji sgn(x)
  3. newData nadpisuje wprowadzane dane
test kończy działanie po przekształceniu danych do formy jednego z wzorców (właściwych/odwrotnych) lub po przekroczeniu max liczby iteracji.
Metody plotPattern i plotWeghts generują pliki graficzne przedstawiające wzorce i macierz wag.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
class hopfieldNet:
    def __init__(self, numOfNeurons): #konstruktor z liczbą neuronów
        self.numOfNeurons = numOfNeurons
        self.weights = np.zeros((numOfNeurons, numOfNeurons)) #macierz wag, początkowo same zera
        self.patterns = [] #wektor z wzorcami
    
    def learn(self, patterns):
        self.patterns = patterns  #lista wzorców trafia do atrybutu patterns
        num_patterns = len(patterns)    
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                if i != j:
                    for pattern in patterns: #uczenie dla wszystkich wzorców
                        self.weights[i][j] += pattern[i] * pattern[j]
                    self.weights[i][j] /= (self.numOfNeurons * num_patterns) #normalizujemy przez liczbę neuronów
    
    def test(self, data, max=10):
    #max odpowiada za maksymalną liczbę iteracji, data wiadomo
        for i in range(max):
            #modyfikacja synchroniczna
            newData = np.dot(self.weights, data) #obliczanie aktywacji każdego neuronu za pomocą iloczynu macierzy
            newData = np.where(newData >= 0, 1, -1) #aktualizacja stanu zgodnie z funkcją aktywacji
            data = newData #nadpisywanie danych
            #jeżeli dane są identyczne jak wzorce (albo ich przeciwności) to kończymy działanie
            for pattern in self.patterns:
                if np.array_equal(data, pattern) or np.array_equal(data, -pattern):
                    print(i+1,data) #testowo funkcja drukuje liczbę iteracji i stan danych
                    return data
        print(i+1,data)
        return data

    def plotWeights(self, title):
        normWeights = (self.weights + 1) / 2 #normalizacja do zakresu [0,1]
        plt.imshow(normWeights)
        plt.savefig(title)

    def plotPatterns(self, imgSize, filename):
        for i in range(len(self.patterns)):
            pattern = self.patterns[i].reshape(imgSize, imgSize) #reshape do kwadratowej macierzy
            plt.imshow(pattern)
            name = filename+f'{i}.png'
            plt.savefig(name)
            plt.close()
