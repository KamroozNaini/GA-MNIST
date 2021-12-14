import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as KL

# load the data
mnist = tf.keras.datasets.mnist  # 28*28 images of hand written digits 0-9
(x_train_s, y_train_s), (x_test_s, y_test_s) = mnist.load_data()

#plt.imshow(x_train_s[0], cmap=plt.cm.binary)
#plt.show()
# end load data

# normalize the data -scaling between 0 and 1 # we do not have to do this
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)
x_train_s, x_test_s = x_train_s / 255.0, x_test_s / 255.0
x_train_s, x_test_s = np.expand_dims(x_train_s, axis=-1), np.expand_dims(x_test_s, axis=-1)
print('x_train shape is ', x_train_s.shape)
# end of normalize
x_train = []
y_train = []
x_test = []
y_test = []
i = 0
print(x_train_s)
print(y_train_s.shape)
x_train = x_train_s[:100]
y_train = y_train_s[:100]
print(x_train)
print(y_train.shape)


# author : Kamrooz Naini
class CNN(Sequential):
    def __init__(self, nfilters, sfilters):
        super().__init__()
        tf.random.set_seed(0)
        self.add(KL.Conv2D(nfilters[0], kernel_size=(sfilters[0], sfilters[0]), padding='same', activation='relu',
                           input_shape=(28, 28, 1)))
        self.add(KL.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(KL.Conv2D(nfilters[1], kernel_size=(sfilters[1], sfilters[1]), padding='same', activation='relu'))
        self.add(KL.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(KL.Conv2D(nfilters[2], kernel_size=(sfilters[2], sfilters[2]), padding='same', activation='relu'))
        self.add(KL.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(KL.Flatten())
        self.add(KL.Dense(10, activation='softmax'))
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class Genetic:

    def __init__(self, pop_size, nlayers, max_nfilters, max_sfilters):
        self.pop_size = pop_size
        self.nlayers = nlayers
        self.max_nfilters = max_nfilters
        self.max_sfilters = max_sfilters
        self.best_acc = 0
        self.best_arch = np.zeros((1, 6))
        self.gen_acc = []

    def generate_population(self):
        np.random.seed(0)
        pop_nlayers = np.random.randint(1, self.max_nfilters, (self.pop_size, self.nlayers))
        pop_sfilters = np.random.randint(1, self.max_sfilters, (self.pop_size, self.nlayers))
        pop_total = np.concatenate((pop_nlayers, pop_sfilters), axis=1)
        return pop_total

    def select_parents(self, pop, nparents, fitness):
        parents = np.zeros((nparents, pop.shape[1]))
        for i in range(nparents):
            best = np.argmax(fitness)
            parents[i] = pop[best]
            fitness[best] = -99999
        return parents

    def crossover(self, parents):
        nchild = self.pop_size - parents.shape[0]
        nparents = parents.shape[0]
        child = np.zeros((nchild, parents.shape[1]))
        for i in range(nchild):
            first = i % nparents
            second = (i + 1) % nparents
            child[i, :2] = parents[first][:2]
            child[i, 2] = parents[second][2]
            child[i, 3:5] = parents[first][3:5]
            child[i, 5] = parents[second][5]
        return child

    def mutation(self, child):
        for i in range(child.shape[0]):
            val = np.random.randint(1, 6)
            ind = np.random.randint(1, 4) - 1
            if child[i][ind] + val > 100:
                child[i][ind] -= val
            else:
                child[i][ind] += val
            val = np.random.randint(1, 4)
            ind = np.random.randint(4, 7) - 1
            if child[i][ind] + val > 20:
                child[i][ind] -= val
            else:
                child[i][ind] += val
        return child

    def fitness(self, pop, X, Y, epochs):
        pop_acc = []
        for i in range(pop.shape[0]):
            nfilters = pop[i][0:3]
            sfilters = pop[i][3:]
            model = CNN(nfilters, sfilters)
            H = model.fit(x_train, y_train, epochs=epochs)
            acc = H.history['accuracy']
            pop_acc.append(max(acc) * 100)
        if max(pop_acc) > self.best_acc:
            self.best_acc = max(pop_acc)
            self.best_chromose = pop[np.argmax(pop_acc)]
        self.gen_acc.append(max(pop_acc))
        return pop_acc


# Main Body
pop_size = 10
nlayers = 3
max_nfilters = 100
max_sfilters = 20
epochs = 5
num_generations = 10
genCNN = Genetic(pop_size, nlayers, max_nfilters, max_sfilters)
pop = genCNN.generate_population()
for i in range(num_generations + 1):
    pop_acc = genCNN.fitness(pop, x_train, y_train, epochs)
    print('Best Accuracy at the generation {}: {}'.format(i, genCNN.best_acc))
    parents = genCNN.select_parents(pop, 5, pop_acc.copy())
    child = genCNN.crossover(parents)
    child = genCNN.mutation(child)
    pop = np.concatenate((parents, child), axis=0).astype('int')
    print("population so far :", pop)
    print("Generation Accuracy :", genCNN.gen_acc)
    print("number of generation ran :", i)
    print("generation best arch", genCNN.best_chromose)
    print('we will calculate the fitness for this generation next round of the loop')
    plt.plot(range(i + 1), genCNN.gen_acc, 'g')
    plt.axis([0, 12, 50, 100])

plt.title('Fitness Accuracy vs Generations')
plt.xlabel('Generations')
plt.ylabel('Fitness (%)')
plt.show()

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))
