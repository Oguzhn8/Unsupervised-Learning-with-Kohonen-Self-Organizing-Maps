import numpy as np
import random

#datayı yükleyelim
f = open("iris.data", "r")
data = f.read()
data_list = data.split("\n")

seperated = []
virginica = []
versicolor = []
setosa = []
#sınıfları belirtelim
for i in range(len(data_list)):
    seperated = data_list[i].split(",")
    sl = float(seperated[0])
    sw = float(seperated[1])
    pl = float(seperated[2])
    pw = float(seperated[3])

    if seperated[4] == "Iris-virginica":
        virginica.append([[sl, sw, pl, pw], 0])
    elif seperated[4] == "Iris-versicolor":
        versicolor.append([[sl, sw, pl, pw], 1])
    elif seperated[4] == "Iris-setosa":
        setosa.append([[sl, sw, pl, pw], 2])
#karıştıralım
random.shuffle(virginica)
random.shuffle(versicolor)
random.shuffle(setosa)
#test ve eğitim kümelerini oluşturalım
train_set = np.concatenate((setosa[:40], np.concatenate((virginica[:40], setosa[:40]))))
test_set = np.concatenate((setosa[40:], np.concatenate((virginica[40:], setosa[40:]))))

grid_1 = 5 
grid_2 = 4
dim = 4

weight_size = grid_1 * grid_2
weights = (np.random.rand(weight_size, dim) - 0.5) * 5  

index_map = np.zeros((weight_size, 2))
for i in range(grid_1):
    for j in range(grid_2):
        index_map[i*grid_2 + j, :] = [i, j]

def sigma (k,sigma_z=2,sigma_o=100):
    return sigma_z * np.exp(-k / sigma_o)

def neighbor(k, i): #i:kazanan nöron    kazanan nöronla diğer nöronların mesafelerini h'a atayalım
    scaling = 2 * np.square(sigma(k))
    h = np.zeros(len(index_map))
    for j in range(len(index_map)):
        neuron_dist = np.linalg.norm(index_map[i, :]-index_map[j, :])
        h[j] = np.exp(-1 * neuron_dist / scaling)
    return h

max_iterations = 1000
learning_rate = 0.005
old_weights = np.zeros(weights.shape)
#eğitim özdüzenleme
for k in range(max_iterations):
    print(k)
    random.shuffle(train_set)
    
    for x, y in train_set:
        d = np.zeros(weight_size)
        for i in range(weight_size):
            d[i] = np.linalg.norm(x-weights[i, :])  #nokta ile nöronun normu
            
        winner_index = np.argmin(d)                  #kazanan nöron
        h = neighbor(k, winner_index)
        
        for j in range(weight_size):                ##ağırlık güncellemesi
            old_weights[j] = weights[j]
            delta_w = learning_rate * h[j] * (x - weights[j])
            
            weights[j] += delta_w 
    
    
        
    if(np.linalg.norm(weights - old_weights).max() < 0.001):    #durdurma kriteri   
        print("Egitim basarili")
        for k in range(grid_1*grid_2*500):                      #yakınsama aşaması
            random.shuffle(train_set)
            for x, y in train_set:
                d = np.zeros(weight_size)
                for i in range(weight_size):
                    d[i] = np.linalg.norm(x-weights[i, :])
                    
                winner_index = np.argmin(d)
                h = neighbor(k, winner_index)
                for j in range(weight_size):                ##ağırlık güncellemesi
                    old_weights[j] = weights[j]
                    delta_w = learning_rate * h[j] * (x - weights[j])
                  #if delta_w.all() == [0 ,0, 0, 0]:
                        #print("Agırlıklar guncellemeyi bırakiyor")
                        #break 
                    
                    weights[j] += delta_w
        break            
#test aşaması   
error = 0
error_mean = 0
for x, y in test_set:
    d = np.zeros(weight_size)
    
    for i in range(weight_size):
        d[i] = np.linalg.norm(x-weights[i, :])
        
        winner_index = np.argmin(d)
        error += d[winner_index] **2 
        error_mean += abs(d[winner_index])
        
print("Karesel Hata:",np.sqrt(error/len(test_set)))
print("Mutlak Hata:",error_mean/len(test_set))
        

