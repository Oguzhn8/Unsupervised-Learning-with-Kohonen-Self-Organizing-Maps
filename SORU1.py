import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
start = datetime.now()

size = 150
variance = 0.5
dim = 3
#eğitim kümesi
X_train = np.concatenate((np.random.multivariate_normal([5, 5, 5], variance*np.eye(dim), size), 
                        np.random.multivariate_normal([5, 0, -5], variance*np.eye(dim), size), 
                        np.random.multivariate_normal([-5, -5, -5], variance*np.eye(dim), size),
                        np.random.multivariate_normal([-5, 0, 5], variance*np.eye(dim), size)))
#nöron sayısı
grid_1 = 5 
grid_2 = 4
#başlangı ağırlıkları
weight_size = grid_1 * grid_2
weights = (np.random.rand(weight_size, dim) - 0.5) * 2

#nöron sınıfı
neuron_class = 4 * np.ones(weight_size)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='k', marker='^')
#plt.show()
#nöron konumları
index_map = np.zeros((weight_size, 2))
for i in range(grid_1):
    for j in range(grid_2):
        index_map[i*grid_2 + j, :] = [i, j]



y_train = np.concatenate((np.zeros(size), np.ones(size), 2*np.ones(size), 3*np.ones(size))) #sınıf bilgileri
#sınıf bilgileri data birleşimi
dataset = list(zip(X_train, y_train))
random.shuffle(dataset) #karıştırma
#eğitim ve test kümelerine ayırma
train_set = dataset[:450]
test_set = dataset[450:]

def sigma (k,sigma_z=2,sigma_o=100):
    return sigma_z * math.exp(-k / sigma_o)

h_grid = np.zeros((grid_1, grid_2))

def neighbor(k, i): #i:kazanan nöron    kazanan nöronla diğer nöronların mesafelerini h'a atayalım
    scaling = 2 * np.square(sigma(k))
    h = np.zeros(len(index_map))
    for j in range(len(index_map)):
        neuron_dist = np.linalg.norm(index_map[i, :]-index_map[j, :])
        h[j] = np.exp(-1 * neuron_dist / scaling)
    return h

max_iterations = 1000   #iterasyon sayısı
learning_rate = 0.05    #öğrenme hızı
old_weights = np.zeros(weights.shape)   #bir önceki adım ağırlıkları

#eğitim süreci özdüzenleme
for k in range(max_iterations):
    
    neuron_class = 4 * np.ones(weight_size)
    
    random.shuffle(train_set)
    error = 0
    for x, y in train_set:
        d = np.zeros(weight_size)
        for i in range(weight_size):
            d[i] = np.linalg.norm(x-weights[i, :])  #nokta ile nöronun normu
            
        winner_index = np.argmin(d)                 #kazanan nöron
        error += d[winner_index]**2
        
        neuron_class[winner_index] = y              #nöronları kazanan index'e yaklaştıralım
        h = neighbor(k, winner_index)               
        for j in range(weight_size):                #ağırlık güncellemesi
            old_weights[j] = weights[j]
            delta_w = learning_rate * h[j] * (x - weights[j])
            weights[j] += delta_w 
    
    
        
    if(np.linalg.norm(weights - old_weights).mean() < 0.01):    #durdurma kriteri
        print("Egitim basarili")
        print(k)
        for k in range(grid_1*grid_2*500):  #yakınsama aşaması
            print(k)
            neuron_class = 4 * np.ones(weight_size)
            
            random.shuffle(train_set)
            for x, y in train_set:
                d = np.zeros(weight_size)
                for i in range(weight_size):
                    d[i] = np.linalg.norm(x-weights[i, :])
                    
                winner_index = np.argmin(d)
                
                neuron_class[winner_index] = y      #nöronları kazanan index'e yaklaştıralım
                h = neighbor(k, winner_index)
                for j in range(weight_size):    #ağırlık güncellemesi
                    old_weights[j] = weights[j]
                    delta_w = learning_rate * h[j] * (x - weights[j])
                    weights[j] += delta_w
        break            
#test aşaması    
error = 0
error_meanabs = 0
for x, y in test_set:
    d = np.zeros(weight_size)
    neuron_class = 4 * np.ones(weight_size)
    for i in range(weight_size):
        d[i] = np.linalg.norm(x-weights[i, :])
        
        winner_index = np.argmin(d)
        error_meanabs += abs(d[winner_index])
        error += d[winner_index]**2
        neuron_class[winner_index] = y
        
print("RMS:",np.sqrt(error/len(test_set)))
print("Mean Absolute Error:", error_meanabs / len(test_set) )
print(datetime.now()- start)

        

