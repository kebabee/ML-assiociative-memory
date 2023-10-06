import numpy as np
import random

class hopfieldNet:
    def __init__(self, numOfNeurons): #konstruktor z liczbą neuronów
        self.numOfNeurons = numOfNeurons
        self.weights = np.zeros((numOfNeurons, numOfNeurons)) #macierz wag, początkowo same zera
        self.patterns = [] #wektor z wzorcem
    
    def learn(self, patterns):
        self.patterns = patterns  # Przypisz listę wzorców do atrybutu patterns
        num_patterns = len(patterns)
    
        # wyzerowanie macierzy wag
        self.weights = np.zeros((self.numOfNeurons, self.numOfNeurons))
    
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                if i != j:
                    for pattern in patterns: #uczenie dla wszystkich wzorców
                        self.weights[i][j] += pattern[i] * pattern[j]
                    self.weights[i][j] /= (self.numOfNeurons * num_patterns) #normalizujemy przez liczbę neuronów
                    #if self.weights[i][j] > 0: #normalizujemy do 1 lub -1
                    #    self.weights[i][j] = 1
                    #elif self.weights[i][j] < 0:
                    #    self.weights[i][j] = -1
    
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
            #jeżeli dane są identyczne jak wzorce albo przeciwności wzorca to kończymy działanie
            for pattern in self.patterns:
                if np.array_equal(data, pattern) or np.array_equal(data, -pattern):
                    return data
        return data

#Część testowa
pattern1 = np.random.choice([-1, 1], size=40)
pattern2 = np.random.choice([-1, 1], size=40)
pattern3 = np.random.choice([-1, 1], size=40)
patterns = [pattern1, pattern2, pattern3]
    
# Tworzenie i uczenie sieci
network = hopfieldNet(40)
network.learn(patterns)
print(network.patterns)
print("Macierz wag:")
for row in network.weights:
    formatted_row = [f"{value:.2f}" for value in row]
    print(formatted_row)
    
# Testowanie sieci z losowymi danymi
testData1 = np.random.choice([-1, 1], size=40)
testData2 = np.random.choice([-1, 1], size=40)
testData3 = np.random.choice([-1, 1], size=40)
print("\nTest1")
result1 = network.test(testData1, 50)
print("\nTest2")
result2 = network.test(testData2, 50)
print("\nTest3")
result3 = network.test(testData3, 50)
