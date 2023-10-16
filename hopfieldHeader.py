"""
Plik nagłówkowy implementujący klasę hopfieldNet.
Klasa zawiera konstruktor tworzący obiekt zawierający liczbę neuronów, macierz wag i macierz z wyuczonymi wzorcami.
Metoda learn implementuje uczenie sieci z wykorzystaniem iloczynu macierzy dla każdego przyjętego wzorca.
Metoda test implementuje działanie sieci zgodnie z modyfikacją synchroniczną, tzn:
  1. wektor NewData początkowo przechowuje aktywacje wszystkich neuronów obliczone za pomoca iloczynu macierzy
  2. każdy element wektora jest przekszałcany za pomocą funkcji sgn(x)
  3. newData nadpisuje wprowadzane dane
test kończy działanie po ustabilizowaniu lub po przekroczeniu max liczby iteracji.
Metoda optimLearn działą zgodnie z opisem na overleafie.
Metody plotPattern i plotWeghts generują pliki graficzne przedstawiające wzorce i macierz wag.
"""
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

class hopfieldNet:
    def __init__(self, numOfNeurons): # konstruktor
        self.numOfNeurons = numOfNeurons
        self.weights = np.random.uniform(-1,1,(numOfNeurons, numOfNeurons)) #macierz wag, początkowo lososwe wartości
        np.fill_diagonal(self.weights, 0) # w_ii = 0
        self.patterns = [] # wektor z wzorcami
    
    def learn(self, patterns): # uczenie metodą Hebba
        self.patterns = patterns
        num_patterns = len(patterns)    
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                if i != j:
                    for pattern in patterns: # dla wszystkich wzorców
                        self.weights[i][j] += pattern[i] * pattern[j]
                    self.weights[i][j] /= (self.numOfNeurons * num_patterns) # normalizujemy przez liczbę neuronów i wzorców
    
    def test(self, data, max=10): # modyfikacja synchroniczna
    #max odpowiada za maksymalną liczbę iteracji
        for i in range(max):
            newData = np.dot(self.weights, data) # obliczanie aktywacji każdego neuronu
            newData = np.where(newData >= 0, 1, -1) # aktualizacja stanu zgodnie z sgn(x)

            energy = self.calculateEnergy(newData)
            print(f'Iteracja {i}, Energia: {energy}') 

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if (np.array_equal(data,newData)):
                print(f'Zakończono po {i} iteracjach (stabilizacja sieci)')
                return newData

            data = newData #nadpisywanie danych
        print(f'Zakończono po {i} iteracjach (przekroczenie zakresu)')
        return data
    
    def asynTest(self, data, max=10): # modyfikacja asynchroniczna
        for i in range(max):
            newData = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            
            for index in indices: # aktualizacja w losowej kolejności
                activation = np.dot(self.weights[index], newData)
                newData[index] = 1 if activation >= 0 else -1 

            energy = self.calculateEnergy(newData)
            print(f'Iteracja {i}, Energia: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, newData):
                print(f'Zakończono po {i} iteracjach (stabilizacja sieci)')
                return newData
            
            data = newData  # nadpisanie danych
        
        print(f'Zakończono po {i} iteracjach (przekroczenie zakresu)')
        return data
    
    def randAsynTest(self, data, max=10): # modyfikacja asynchroniczna "nienadzorowana"
        for i in range(max):
            newData = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów

            # aktualizacja neuronów w losowej kolejności
            for _ in range(len(data)):
                index = random.choice(indices) # dany indeks może zostać wybrany wielokrotnie lub wcale
                activation = np.dot(self.weights[index], newData)  # aktywacja jednego neuronu
                newData[index] = 1 if activation >= 0 else -1  # aktualizacja stan neuronu

            energy = self.calculateEnergy(newData)
            print(f'Iteracja {i}, Energia: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, newData):
                print(f'Zakończono po {i} iteracjach (stabilizacja sieci)')
                return newData

            data = newData  # nadpisanie danych

        print(f'Zakończono po {i} iteracjach (przekroczenie zakresu)')
        return data

    def plotWeights(self, title): # plot macierzy wag
        normWeights = (self.weights + 1) / 2 #normalizacja do zakresu [0,1]
        plt.imshow(normWeights, cmap='gray')
        plt.savefig(f"wyniki1/"+title)
        # plt.show()

    def plotPatterns(self, imgSize, filename): # plot wzorców
        for i in range(len(self.patterns)):
            pattern = self.patterns[i].reshape(imgSize, imgSize) #reshape do kwadratowej macierzy
            plt.imshow(pattern, cmap='gray')
            name = filename+f'{i}.png'
            plt.savefig(f"wyniki1/"+name)
            # plt.show()
            plt.close()

    def calculateEnergy(self, state):
        energy = -0.5*np.dot(np.dot(np.transpose(state),self.weights),state)
        return energy
    
    def addPatterns(self, patterns):
        self.patterns = patterns

    def optimLearn6(self, pattern, learningRate):
        evolPattern = self.asynTest(pattern)

        self.weights += learningRate*(np.outer(pattern, pattern) - np.outer(evolPattern, evolPattern))
        np.fill_diagonal(self.weights, 0)

def noise(data, changes=5): # funkcja zniekształcająca dane
    tempData = data.copy()
    randomIndexes = random.sample(range(len(data)), changes)

    for index in randomIndexes:
        tempData[index] = -tempData[index]

    return tempData
