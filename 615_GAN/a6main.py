import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from Activation import ReLu
from Activation import ReLuTest
from FullyConnected import FullyConnected
from Activation import Sigmoid
from Output import LogLoss
from Output import Generator


def createStochasticBatch(sourceBatch, batchSize):
    np.random.shuffle(sourceBatch)
    newArr = sourceBatch[:batchSize]
    return newArr


def createInput(batchSize):
    input = np.random.randint(256, size=(batchSize, 784))
    return input


def createImage(arr, title):
    # print("\nShow best image")
    arr = arr.reshape(28, 28)
    plt.imshow(arr, cmap=cm.gray)
    plt.axis('off')
    # plt.savefig("OutFiles\Part2-" + title + ".png")
    plt.show()


def combineRealandFake(arr1, arr2):
    arrOut = np.append(arr1, arr2, axis=0)
    return arrOut


def ReluGAN(showEachEpoch):
    np.random.seed(0)
    batchSize = 100
    numClasses = 10

    # Setup starting arrays
    print("Reading training data. Please wait...")
    trainArr = pd.read_csv("mnist_train.csv", header=None)

    print("\nSort by first column...")
    trainArr = trainArr.sort_values(by=[0])

    print("\nSort by first column...")
    trainArr = trainArr.to_numpy()

    print("\nSet up arrays for each class...")
    indexList = np.array([], dtype=int)

    currentClass = trainArr[0, 0]
    numObservations = trainArr.shape[0]
    for j in range(numObservations):
        if trainArr[j, 0] != currentClass:
            indexList = np.append(indexList, int(j))
            currentClass = trainArr[j, 0]

    trainArr = trainArr[:, 1:]  # Remove the first column before splitting the data
    arrList = np.split(trainArr, indexList, axis=0)  # now the index of the array represents its class

    print("\nSet up target arrays based on batch size...")
    d_trainArrTarget = np.ones((batchSize, 1), dtype=int)
    d_fakeArrTarget = np.zeros((batchSize, 1), dtype=int)
    targetArr_d = combineRealandFake(d_trainArrTarget, d_fakeArrTarget)

    print("\nTraining...")
    for i in range(10):
        print("Index " + str(i) + "...")
        # TODO:
        learningRate = 0.0001
        LR = 0.0001

        # Generator
        FC_G = FullyConnected(784, 784, LR)
        relu_G = ReLu()
        objective_G = Generator()

        # Discriminator
        FC_D = FullyConnected(784, 1, learningRate)
        sig_D = Sigmoid()
        LL_D = LogLoss(targetArr_d)

        # parameters
        epoch = 1
        jChange = 100
        # Digit = 0

        batchArr = arrList[i]
        mu = np.average(batchArr)
        sigma = np.std(batchArr)
        input = np.random.normal(mu, sigma, size=(batchSize, 784))

        while jChange > 10 ** -6 and epoch <= 30:
            # print(epoch)
            # Fake input
            input_f = np.random.normal(mu, sigma, size=(batchSize, 784))
            # Real input
            input_r = createStochasticBatch(batchArr, batchSize)

            # Forward Prop
            X_f = FC_G.forwardPropagate(input_f)
            X_f = relu_G.forwardPropagate(X_f)  # Generated fake data

            input_d = combineRealandFake(input_r, X_f)
            # print(input_d)
            X_d = FC_D.forwardPropagate(input_d)
            X_d = sig_D.forwardPropagate(X_d)
            J_d = LL_D.eval(X_d)

            # back prop discriminator
            grad = LL_D.gradient(X_d)
            grad = sig_D.backwardPropagate(grad)
            grad = FC_D.backwardPropagate(grad, epoch)

            # forward prop again
            X_d = FC_D.forwardPropagate(X_f)
            X_d = sig_D.forwardPropagate(X_d)
            J_g = objective_G.eval(X_d)

            # back again
            grad = objective_G.gradient(X_d)
            grad = sig_D.backwardPropagate(grad)
            grad = FC_D.backwardPropagateNoUpdate(grad)
            grad = relu_G.backwardPropagate(grad)
            FC_G.backwardPropagate(grad, epoch)
            # print(np.max(X_d, axis=0))

            epoch += 1

            if showEachEpoch:
                best_index = np.argmax(X_d, axis=0)
                best = X_f[best_index]
                createImage(best, "d" + str(i) + "e" + str(epoch))
        if not showEachEpoch:
            best_index = np.argmax(X_d, axis=0)
            best = X_f[best_index]
            createImage(best, "d" + str(i) + "e" + str(epoch))

def StepGAN(showEachEpoch):
    np.random.seed(0)
    batchSize = 100
    numClasses = 10

    # Setup starting arrays
    print("Reading training data. Please wait...")
    trainArr = pd.read_csv("mnist_train.csv", header=None)
    # print(trainArr)
    print("\nSort by first column...")
    trainArr = trainArr.sort_values(by=[0])
    # print(trainArr)
    print("\nSort by first column...")
    trainArr = trainArr.to_numpy()

    print("\nSet up arrays for each class...")
    arrList = np.array([])
    indexList = np.array([], dtype=int)

    currentClass = trainArr[0, 0]
    numObservations = trainArr.shape[0]
    for j in range(numObservations):
        if trainArr[j, 0] != currentClass:
            indexList = np.append(indexList, int(j))
            currentClass = trainArr[j, 0]

    trainArr = trainArr[:, 1:]  # Remove the first column before splitting the data
    arrList = np.split(trainArr, indexList, axis=0)  # now the index of the array represents its class

    print("\nSet up target arrays based on batch size...")
    d_trainArrTarget = np.ones((batchSize, 1), dtype=int)
    d_fakeArrTarget = np.zeros((batchSize, 1), dtype=int)
    targetArr_d = combineRealandFake(d_trainArrTarget, d_fakeArrTarget)

    print("\nTraining...")
    for i in range(10):
        print("Index " + str(i) + "...")
        # TODO:
        learningRate = 0.0002
        # LR = 0.0001
        LR = 1
        # Generator
        FC_G = FullyConnected(784, 784, LR)
        # relu_G = ReLu()
        relu_G = ReLuTest()
        objective_G = Generator()

        # Discriminator
        FC_D = FullyConnected(784, 1, learningRate)
        # FC_D.printParameters()
        sig_D = Sigmoid()
        LL_D = LogLoss(targetArr_d)


        # parameters
        epoch = 1
        jChange = 100
        # Digit = 0

        batchArr = arrList[i]
        mu = np.average(batchArr)
        sigma = np.std(batchArr)
        input = np.random.normal(mu, sigma, size=(batchSize, 784))

        while jChange > 10 ** -6 and epoch <= 30:
            # print(epoch)
            # Fake input
            input_f = np.random.normal(mu, sigma, size=(batchSize, 784))
            # Real input
            input_r = createStochasticBatch(batchArr, batchSize)

            # Forward Prop
            X_f = FC_G.forwardPropagate(input_f)
            X_f = relu_G.forwardPropagate(X_f)  # Generated fake data

            input_d = combineRealandFake(input_r, X_f)
            # print(input_d)
            X_d = FC_D.forwardPropagate(input_d)
            X_d = sig_D.forwardPropagate(X_d)
            J_d = LL_D.eval(X_d)

            # back prop discriminator
            grad = LL_D.gradient(X_d)
            grad = sig_D.backwardPropagate(grad)
            grad = FC_D.backwardPropagate(grad, epoch)

            # forward prop again
            X_d = FC_D.forwardPropagate(X_f)
            X_d = sig_D.forwardPropagate(X_d)
            J_g = objective_G.eval(X_d)

            # back again
            grad = objective_G.gradient(X_d)
            grad = sig_D.backwardPropagate(grad)
            grad = FC_D.backwardPropagateNoUpdate(grad)
            grad = relu_G.backwardPropagate(grad)
            FC_G.backwardPropagate(grad, epoch)
            # print(np.max(X_d, axis=0))
            epoch += 1

            if showEachEpoch:
                best_index = np.argmax(X_d, axis=0)
                best = X_f[best_index]
                createImage(best, "d" + str(i) + "e" + str(epoch))
        if not showEachEpoch:
            best_index = np.argmax(X_d, axis=0)
            best = X_f[best_index]
            createImage(best, "d" + str(i) + "e" + str(epoch))


showEachEpoch = True
ReluGAN(showEachEpoch)
StepGAN(showEachEpoch)
