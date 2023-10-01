import numpy as np
import random

class hopfieldNet:
    def __init__(self, numOfNeurons): #konstruktor z liczbą neuronów
        self.numOfNeurons = numOfNeurons
        self.weights = np.zeros((numOfNeurons, numOfNeurons)) #macierz wag, początkowo same zera
        self.thresholds = np.zeros(numOfNeurons)
    
    def learn(self, pattern): #zapamiętywanie wzorca
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                if i != j: #wagi połączeń neuronów samych ze sobą pozostają zerowe
                    self.weights[i][j] += pattern[i] * pattern[j]
        self.weights /= self.numOfNeurons #normalizacja przez liczbę neuronów
        self.weights = np.where(self.weights > 0, 1, -1) #normalizacja wag do wartości -1 lub 1
    
    def test(self, data, max=10):
    #max odpowiada za maksymalną liczbę iteracji, data to wprowadzane dane
        for _ in range(max):
            newData = np.dot(self.weights, data)
            newData = np.where(newData >= 0, 1, -1)
            if np.array_equal(newData, data): #jeżeli dane są identyczne jak wzorzec kończymy działanie
                break
            data = newData #nadpisywanie danych
        return data

#test działania
pattern = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    
# Tworzenie i uczenie sieci
network = hopfieldNet(8)
network.learn(pattern)
    
# Testowanie sieci z danymi zakłóconymi
testData = np.array([-1, -1, 1, -1, 1, -1, 1, -1])
result = network.test(testData)


print("Macierz wag:")
print(network.weights)    
print("Odzyskany wzorzec po testowaniu:")
print(result)