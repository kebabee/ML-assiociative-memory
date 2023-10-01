import numpy as np
import random

class hopfieldNet:
    def __init__(self, numOfNeurons): #konstruktor z liczbą neuronów
        self.numOfNeurons = numOfNeurons
        self.weights = np.zeros((numOfNeurons, numOfNeurons)) #macierz wag, początkowo same zera
        self.pattern = np.zeros(numOfNeurons) #wektor z wzorcem
    
    def learn(self, pattern): #zapamiętywanie wzorca i obliczanie wag
        self.pattern = pattern
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                if i != j: #wagi połączeń neuronów samych ze sobą pozostają zerowe
                    self.weights[i][j] = pattern[i] * pattern[j]
                    self.weights[i][j] /= self.numOfNeurons #normalizujemy przez liczbę neuronów
                    #if self.weights[i][j] > 0: #normalizujemy do 1 lub -1
                    #  self.weights[i][j] = 1
                    #elif self.weights[i][j] < 0:
                    #  self.weights[i][j] = -1
    
    def test(self, data, max=10):
    #max odpowiada za maksymalną liczbę iteracji, data wiadomo
        for i in range(max):
            print(f"Numer iteracji: {i+1}")
            print(f"stan przed: {data}")
            #modyfikacja synchroniczna
            newData = np.dot(self.weights, data) #obliczanie aktywacji każdego neuronu za pomocą iloczynu macierzy
            newData = np.where(newData >= 0, 1, -1) #aktualizacja stanu zgodnie z funkcją aktywacji
            print(f"stan po:    {newData}")
            data = newData #nadpisywanie danych
            #jeżeli dane są identyczne jak wzorzec albo przeciwność wzorca to kończymy działanie pętli
            if np.array_equal(data, self.pattern) or np.array_equal(data, -self.pattern):
                break
        return data

#Część testowa
pattern = np.array([1, -1, 1, 1, -1, 1, -1, -1])
print(f"Wzorzec: {pattern}\nWzorzec odwrotny: {-pattern}")
    
# Tworzenie i uczenie sieci
network = hopfieldNet(8)
network.learn(pattern)
    
# Testowanie sieci z danymi zakłóconymi
testData1 = np.array([-1, -1, 1, 1, -1, 1, -1, -1]) #jeden błąd w stosunku do wzorca
testData2 = np.array([-1, -1, 1, 1, 1, 1, -1, 1]) #3 błędy w stodunku do wzorca
testData3 = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #dwa błędy w stosunku do wzorca odwrotnego
testData4 = np.array([1, -1, 1, 1, 1, -1, 1, 1]) #tak samo podobne do właściwego i odwrotnego wzorca
print("\nTest1")
result1 = network.test(testData1) #poprawnie odtworzony wzorzec
print("\nTest2")
result2 = network.test(testData2) #poprawnie odtworzony wzorzec
print("\nTest3")
result3 = network.test(testData3) #poprawnie odtworzony odwrotny wzorzec
print("\nTest4")
result4 = network.test(testData4) #sieci nie udało się ustabilizować

print("\nMacierz wag:")
print(network.weights) 
