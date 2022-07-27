import numpy as np
import pandas as pd
from scipy import ndimage as sci
from matplotlib import pyplot as plt
import random as rand

class network() :

    data = pd.read_csv(r'train.csv')
    data = np.array(data)
    np.random.shuffle(data)


    m, n = data.shape
    dataTest = data[0:1000]
    dataTest = dataTest.T
    YTest = dataTest[0]
    XTest = dataTest[1:n]
    XTest = XTest/255


    dataTrain = data[1000: m]
    dataTrain = dataTrain.T
    YTrain = dataTrain[0]
    XTrain = dataTrain[1:n]
    XTrain = XTrain/255

    XTrain = XTrain.T
    rotated = np.zeros((2 * len(XTrain), len(XTrain[0])), int)

    for i in range(len(XTrain)) :
        testRot = np.array(XTrain[i]).reshape(28,28)
        testRot = sci.rotate(testRot, angle = rand.randint(5, 15), reshape = False, mode = 'nearest')
        rotated[i] = testRot.reshape(1, 784)

    for i in range(len(XTrain)) :
        testRot = np.array(XTrain[i]).reshape(28,28)
        testRot = sci.rotate(testRot, angle = rand.randint(345, 355), reshape = False, mode = 'nearest')
        rotated[i + len(XTrain)] = testRot.reshape(1, 784)


    XTrain = XTrain.T
    rotated = rotated.T

    XTrain = np.concatenate((XTrain, rotated), axis = 1)
    tmp_set = np.concatenate((YTrain, YTrain))
    YTrain = np.concatenate((YTrain, tmp_set))

    def gradientDescent(X, Y, X_test, Y_test, iterations, alpha, batches) :


        def initParams() :
            W1 = np.random.rand(60, 784) - 0.5
            b1 = np.random.rand(60, 1) - 0.5
            W2 = np.random.rand(40, 60) - 0.5
            b2 = np.random.rand(40, 1) - 0.5
            W3 = np.random.rand(20, 40) - 0.5
            b3 = np.random.rand(20, 1) - 0.5
            W4 = np.random.rand(10, 20) - 0.5
            b4 = np.random.rand(10, 1) - 0.5
            return W1, b1, W2, b2, W3, b3, W4, b4

        def forwardProp(W1, b1, W2, b2, W3, b3, W4, b4, X, X_test) :

            def ReLU(Z) :
                return np.maximum(0, Z)

            def softmax(Z) :
                return np.exp(Z) / sum(np.exp(Z))

            Z1 = W1.dot(X) + b1
            A1 = ReLU(Z1)
            Z2 = W2.dot(A1) + b2
            A2 = ReLU(Z2)
            Z3 = W3.dot(A2) + b3
            A3 = ReLU(Z3)
            Z4 = W4.dot(A3) + b4
            A4 = softmax(Z4)

            tmp1 = W1.dot(X_test) + b1
            test1 = ReLU(tmp1)
            tmp2 = W2.dot(test1) + b2
            test2 = ReLU(tmp2)
            tmp3 = W3.dot(test2) + b3
            test3 = ReLU(tmp3)
            tmp4 = W4.dot(test3) + b4
            A_test = softmax(tmp4)
            return Z1, A1, Z2, A2, Z3, A3, Z4, A4, A_test


        def backProp(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W2, W3, W4, Y, X) :

            def initY(Y) :
                one_hot_Y = np.zeros((Y.size, Y.max() + 1))
                one_hot_Y[np.arange(Y.size), Y] = 1
                one_hot_Y = one_hot_Y.T
                return one_hot_Y

            def derivReLU(Z) :
                return Z > 0

            one_hot_Y = initY(Y)
            dZ4 = A4 - one_hot_Y
            dW4 = (1/len(X[0])) * dZ4.dot(A3.T)
            db4 = (1/len(X[0])) * np.sum(dZ4)
            dZ3 = W4.T.dot(dZ4) * derivReLU(Z3)
            dW3 = (1/len(X[0])) * dZ3.dot(A2.T)
            db3 = (1/len(X[0])) * np.sum(dZ3)
            dZ2 = W3.T.dot(dZ3) * derivReLU(Z2)
            dW2 = (1/len(X[0])) * dZ2.dot(A1.T)
            db2 = (1/len(X[0])) * np.sum(dZ2)
            dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
            dW1 = (1/len(X[0])) * dZ1.dot(X.T)
            db1 = (1/len(X[0])) * np.sum(dZ1)
            return dW1, db1, dW2, db2, dW3, db3, dW4, db4

        def updateParams(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha) :
            W1 = W1 - (alpha * dW1)
            b1 = b1 - (alpha * db1)
            W2 = W2 - (alpha * dW2)
            b2 = b2 - (alpha * db2)
            W3 = W3 - (alpha * dW3)
            b3 = b3 - (alpha * db3)
            W4 = W4 - (alpha * dW4)
            b4 = b4 - (alpha * db4)
            return W1, b1, W2, b2, W3, b3, W4, b4

        def getPredictions(A4) :
            return np.argmax(A4, 0)

        def getAccuracy(predictions, Y) :
            return np.sum(predictions == Y) / Y.size

        def alphaDecay(n) :
            return round(1/(n+3), 2)


        W1, b1, W2, b2, W3, b3, W4, b4 = initParams()
        for j in range(iterations) :

            for i in range(batches) :
                Z1, A1, Z2, A2, Z3, A3, Z4, A4, A_test = forwardProp(W1, b1, W2, b2, W3, b3, W4, b4, X[:, (i*12300):(i*12300)+12300], X_test)
                dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backProp(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W2, W3, W4, Y[(i*12300):(i*12300)+12300], X[:, (i*12300):(i*12300)+12300])
                W1, b1, W2, b2, W3, b3, W4, b4 = updateParams(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)

            Z1, A1, Z2, A2, Z3, A3, Z4, A4, A_test = forwardProp(W1, b1, W2, b2, W3, b3, W4, b4, X, X_test)
            dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backProp(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W2, W3, W4, Y, X)
            W1, b1, W2, b2, W3, b3, W4, b4 = updateParams(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)

            if j % 50 == 0 :
                print("Iterations: ", j)
                print("Training accuracy: ", getAccuracy(getPredictions(A4), Y))
                print("Test accuracy: ", getAccuracy(getPredictions(A_test), Y_test))
        return W1, b1, W2, b2, W3, b3, W4, b4

    gradientDescent(XTrain, YTrain, XTest, YTest, 2501, 0.25, int(len(XTrain[0])/12300))
