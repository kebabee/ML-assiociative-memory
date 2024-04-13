import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

def F(x,n):
    return x**n if x >= 0 else 0

class hopfield_net:
    def __init__(self, num_of_neurons): # konstruktor
        self.num_of_neurons = num_of_neurons
        self.weights = np.random.uniform(-1,1,(num_of_neurons, num_of_neurons)) #macierz wag, początkowo lososwe wartości
        np.fill_diagonal(self.weights, 0) # w_ii = 0
        self.patterns = [] # wektor z wzorcami
    
    def learn(self, patterns): # uczenie metodą Hebba
        # self.weights = np.random.uniform(-1,1,(self.numOfNeurons, self.numOfNeurons))
        self.patterns = patterns
        num_patterns = len(patterns)    
        for i in range(self.num_of_neurons):
            for j in range(self.num_of_neurons):
                if i != j:
                    for pattern in patterns: # dla wszystkich wzorców
                        self.weights[i][j] += pattern[i] * pattern[j]
                    self.weights[i][j] /= (self.num_of_neurons * num_patterns) # normalizujemy przez liczbę neuronów i wzorców
    
    def test(self, data, max=10): # modyfikacja synchroniczna
    #max odpowiada za maksymalną liczbę iteracji
        for i in range(max):
            new_data = np.dot(self.weights, data) # obliczanie aktywacji każdego neuronu
            new_data = np.where(new_data >= 0, 1, -1) # aktualizacja stanu zgodnie z sgn(x)

            energy = self.calculate_energy(new_data)
            print(f'Iteration {i}, energy: {energy}') 

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if (np.array_equal(data,new_data)):
                closestPattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closestPattern)
                print(f'Completed in {i} iterations (network stable), error: {error}')
                return new_data

            data = new_data #nadpisywanie danych
        print(f'Completed in {i} iterations (iterations limit)')
        return data
    
    def asyn_test(self, data, max=10): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            
            for index in indices: # aktualizacja w losowej kolejności
                activation = np.dot(self.weights[index], new_data)
                new_data[index] = 1 if activation >= 0 else -1 

            energy = self.calculate_energy(new_data)
            print(f'Iteration {i}, energia: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'Completed in {i} iterations (network stable), error: {error}')
                return new_data
            
            data = new_data  # nadpisanie danych
        
        print(f'Completed in {i} iterations (iteration limit))')
        return data
    
    def asyn_test2(self, data, max=10): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            
            for index in indices: # aktualizacja w losowej kolejności
                activation = np.dot(self.weights[index], new_data)
                new_data[index] = 1 if activation >= 0 else -1 

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'{error},')
                return new_data
            
            data = new_data  # nadpisanie danych
        
        closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
        error = np.sum(new_data != closest_pattern)
        print(f'{error},')
        return data
    
    def asyn_test_tanh(self, data, max=10, T=1): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            
            for index in indices: # aktualizacja w losowej kolejności
                activation = np.dot(self.weights[index], new_data)
                new_data[index] = np.tanh(T*activation)

            energy = self.calculate_energy(new_data)
            print(f'Iteration {i}, energy: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'Completed in {i} iterations (network stable), error: {error}')
                return new_data
            
            data = new_data  # nadpisanie danych
        
        print(f'Completed in {i} iterations (iterations limit)')
        return data
        
    def rand_asyn_test(self, data, max=10): # modyfikacja asynchroniczna "nienadzorowana"
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów

            # aktualizacja neuronów w losowej kolejności
            for _ in range(len(data)):
                index = random.choice(indices) # dany indeks może zostać wybrany wielokrotnie lub wcale
                activation = np.dot(self.weights[index], new_data)  # aktywacja jednego neuronu
                new_data[index] = 1 if activation >= 0 else -1  # aktualizacja stan neuronu

            energy = self.calculate_energy(new_data)
            print(f'Iteration {i}, energy: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'Completed in {i} iterations (network stable), error: {error}')
                return new_data

            data = new_data  # nadpisanie danych

        print(f'Completed in {i} iterations (iterations limit))')
        return data

    def dense_mem_test(self, data, max=10, n=5): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            x,z = np.shape(self.patterns)
            for index in indices:
                activation = 0
                for mu in range(x):
                    sum_value = 0
                    for j in range(len(indices)):
                        sum_value += self.patterns[mu][j] * data[j]
                    sum_value -= self.patterns[mu][index] * data[index]
        
                    activation += F(self.patterns[mu][index] + sum_value, n) - F(-self.patterns[mu][index] + sum_value, n)

                new_data[index] = 1 if activation >= 0 else -1 

            energy = self.calculate_energy(new_data)
            print(f'Iteration {i}, energy: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'Completed in {i} iterations (network stable), error: {error}')
                return new_data
            
            data = new_data  # nadpisanie danych
        
        print(f'Completed in {i} iterations (iteration limit)')
        return data
    
    def dense_mem_test2(self, data, max=10, n=5): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            x,z = np.shape(self.patterns)
            for index in indices:
                activation = 0
                for mu in range(x):
                    sum_value = 0
                    for j in range(len(indices)):
                        sum_value += self.patterns[mu][j] * data[j]
                    sum_value -= self.patterns[mu][index] * data[index]
        
                    activation += F(self.patterns[mu][index] + sum_value, n) - F(-self.patterns[mu][index] + sum_value, n)

                new_data[index] = 1 if activation >= 0 else -1 

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closest_pattern)
                print(f'{error},')
                return new_data
            
            data = new_data  # nadpisanie danych
        
        closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
        error = np.sum(new_data != closest_pattern)
        print(f'{error},')
        return data
    
    # def hopfieldLayer(self, data, beta = 1):
    #     XT = np.transpose(self.patterns)
    #     softmax_input = beta * data @ XT - np.max(beta * data @ XT)
    #     softmax_result = np.exp(softmax_input) / np.sum(np.exp(softmax_input), axis=0, keepdims=True)
    #     newData = softmax_result @ self.patterns

    #     closestIndex, closestPattern = min(enumerate(self.patterns), key=lambda x: np.sum(x[1] != newData))
    #     # error = np.sum(newData != closestPattern)
    #     # print(f'Błąd: {error}, dopasowanie: {closestIndex}')

    #     return newData, closestIndex

    def hopfield_layer(self, data, beta=1):
        XT = np.transpose(self.patterns)

        # Subtracting the maximum value to avoid overflow
        softmax_input = beta * (data @ XT) - np.max(beta * data @ XT)

        # Calculate softmax function with numerical stability
        exp_values = np.exp(softmax_input)
        softmax_result = exp_values / np.sum(exp_values, axis=0, keepdims=True)

        new_data = softmax_result @ self.patterns

        closest_index, closest_pattern = min(enumerate(self.patterns), key=lambda x: np.sum(x[1] != new_data))
        # error = np.sum(newData != closestPattern)
        # print(f'Błąd: {error}, dopasowanie: {closestIndex}')
        return new_data, closest_index
    
    def hopfield_layer2(self, data, beta=5):
        XT = np.transpose(self.patterns)

        # Subtracting the maximum value to avoid overflow
        softmax_input = beta * data @ XT - np.max(beta * data @ XT)

        # Calculate softmax function with numerical stability
        exp_values = np.exp(softmax_input)
        softmax_result = exp_values / np.sum(exp_values, axis=0, keepdims=True)

        new_data = softmax_result @ self.patterns

        closest_pattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
        error = np.sum(new_data != closest_pattern)
        print(f'{error},')
        return new_data

    def plot_weights(self, title): # plot macierzy wag
        norm_weights = (self.weights + 1) / 2 #normalizacja do zakresu [0,1]
        plt.imshow(norm_weights, cmap='grey')
        plt.savefig(f"shit/"+title)
        # plt.show()

    def plot_patterns(self, height, width, filename): # plot wzorców
        for i in range(len(self.patterns)):
            pattern = self.patterns[i].reshape(height, width) #reshape do kwadratowej macierzy
            plt.imshow(pattern, cmap='inferno')
            name = filename+f'{i}.png'
            plt.grid(False)
            plt.axis('off')
            # plt.savefig(f"wyniki6/"+name)
            plt.show()
            plt.close()

    def calculate_energy(self, state):
        energy = -0.5*np.dot(np.dot(np.transpose(state),self.weights),state)
        return energy
    
    def set_patterns(self, patterns):
        self.patterns = patterns

    def optim_learn6(self, pattern, learning_rate):
        evol_pattern = self.asyn_test(pattern)

        self.weights += learning_rate*(np.outer(pattern, pattern) - np.outer(evol_pattern, evol_pattern))
        np.fill_diagonal(self.weights, 0)

    def reversed_optim_learn6(self, pattern, learning_rate):
        evol_pattern = self.asyn_test(pattern)

        self.weights += learning_rate*(- np.outer(pattern, pattern) + np.outer(evol_pattern, evol_pattern))
        np.fill_diagonal(self.weights, 0)

    def gif_asyn_test(self, data, max=10, name="eeeee"): # modyfikacja asynchroniczna
        for i in range(max):
            new_data = np.copy(data)
            indices = list(range(len(data)))  # lista indeksów neuronów            
            # losowo permutujemy indeksy żeby aktualizować w losowej kolejności
            random.shuffle(indices)
            
            for index in indices: # aktualizacja w losowej kolejności
                activation = np.dot(self.weights[index], new_data)
                new_data[index] = 1 if activation >= 0 else -1 

            plt.imshow(data.reshape(16, 16), cmap='gray')
            plt.savefig(f'prezentacjaGif/{name}-{i}.png')
            # plt.show()

            energy = self.calculate_energy(new_data)
            print(f'Iteracja {i}, Energia: {energy}')

            # jeśli stan sieci nie ulega zmianie to kończymy działanie
            if np.array_equal(data, new_data):
                plt.imshow(data.reshape(16, 16), cmap='gray')
                plt.savefig(f'prezentacjaGif/{name}-{i}.png')
                closestPattern = min(self.patterns, key=lambda pattern: np.sum(pattern != new_data))
                error = np.sum(new_data != closestPattern)
                print(f'Zakończono po {i} iteracjach (stabilizacja sieci), błąd: {error}')
                return new_data
            
            plt.imshow(data.reshape(16, 16), cmap='gray')
            plt.savefig(f'prezentacjaGif/{name}-{i}.png')

            data = new_data  # nadpisanie danych
        
        print(f'Zakończono po {i} iteracjach (przekroczenie zakresu)')
        return data        

def noise(data, changes=5): # funkcja zniekształcająca dane
    temp_data = data.copy()
    random_indexes = random.sample(range(len(data)), changes)

    for index in random_indexes:
        temp_data[index] = -temp_data[index]

    return temp_data

def cut(data, changes=1):
    temp_data = data.copy()
    for i in range(changes):
        temp_data[i] = 0
    return temp_data
