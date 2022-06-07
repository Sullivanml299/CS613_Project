from math import floor, ceil
import numpy as np
from scipy.sparse import csr_matrix

class Conv2DTranspose():
    def __init__(self, channels, W, padding="same", strides=(1, 1)):
        self.X = None
        self.channels = channels
        self.W = W
        self.padding = padding
        self.strides = strides

    def forwardPropagate(self, dataIn):
        self.X = dataIn

        # Define output shape
        if(self.W.ndim == 3):
            channels, kernelHeight, kernelWidth = self.W.shape
        else:
            kernelHeight, kernelWidth = self.W.shape

        inputRows = self.X.shape[0]
        inputCols = self.X.shape[1]

        outputChannels = self.channels
        outputRows = inputRows + kernelHeight - 1
        outputCols = inputCols + kernelWidth - 1
        if outputChannels > 1:
            output = np.zeros([outputChannels, outputRows, outputCols])
        else:
            output = np.zeros([outputRows, outputCols])
        # print("Output: \n {}".format(output))

        # Generate Sparse Convolutional Matrix
        matrixShape = (inputCols * inputRows, outputRows * outputCols)
        print("shape", matrixShape)
        sparseMatrix = np.zeros(matrixShape)
        print(sparseMatrix)

        # build kernel vector that will be shifted across the sparse matrix
        # Note that since this is Transpose, everything is based on the output size
        numKernelValues = kernelWidth*kernelHeight
        numZerosBetweenRows = (outputCols - kernelWidth)
        numRows = kernelHeight - 1
        kernelVector = np.zeros(numKernelValues + numZerosBetweenRows * numRows)
        print(kernelVector)
        for row in range(kernelHeight):
            start = row * kernelWidth + row * numZerosBetweenRows
            end = start + kernelWidth
            # print(start)
            kernelVector[start:end] = self.W[row]
            # print(kernelVector)

        # Fill the sparse Matrix
        numHorizontalStrides = int(outputCols - kernelWidth/self.strides[1])

        h_strides = 0
        v_strides = 0
        for i in range(matrixShape[0]):
            # print("h_strides:", h_strides, "v_strides:", v_strides)
            start = v_strides * outputCols + h_strides
            end = start + kernelVector.shape[0]
            # print("Start:", start, "End:", end)
            sparseMatrix[i, start:end] = kernelVector

            # update strides
            h_strides += 1
            if(h_strides > numHorizontalStrides):
                h_strides = 0
                v_strides += 1
        # print(sparseMatrix)

        w = sparseMatrix #self.kernel2matrix(self.W, (outputRows, outputCols))
        print("w: \n {}".format(w.shape))
        # # print("scipy: \n {}".format(csr_matrix(self.W, shape=(outputRows, outputCols)).toarray()))
        # print("X: \n {}".format(self.X))
        print("Output: \n {}".format((w.T @ self.X.reshape(-1)).reshape(outputRows, outputCols)))


        # Calculate the output
        for c in range(outputChannels):
            # print("Channel: \n {}".format(self.W[c, :, :]))
            for i in range(inputRows):
                for j in range(inputCols):
                    if outputChannels > 1:
                        out = self.X[i, j] * self.W[c, :, :]
                        output[c, i: i + kernelHeight, j: j + kernelWidth] += out
                    else:
                        out = self.X[i, j] * self.W
                        output[i: i + kernelHeight, j: j + kernelWidth] += out
            # print("Output: \n {}".format(output))

        # Add padding
        if self.padding == "same":
            if outputChannels > 1:
                output = output[:, :inputRows, :inputCols]
            else:
                output = output[:inputRows, :inputCols]

        return output

    def kernel2matrix(self, K, outputShape):
        wRows = np.product(K.shape)
        wCols = np.product(outputShape)
        k, W = np.zeros(5), np.zeros((wRows, wCols))
        k[:2], k[3:5] = K[0, :], K[1, :]
        W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
        return W

class Conv2D():
    def __init__(self, kernel, stride=(1, 1)):
        self.kernel = kernel
        self.stride = stride

    def forwardPropagate(self, dataIn):
        kernelHeight = self.kernel.shape[0]
        kernelWidth = self.kernel.shape[1]

        verticalStrideSize = self.stride[0]
        horizontalStrideSize = self.stride[1]

        inputHeight = dataIn.shape[0]
        inputWidth = dataIn.shape[1]

        # print("Kernel Height:", kernelHeight)
        # print("Kernel Width:", kernelWidth)
        #
        # print("Data Height:", dataIn.shape[0])
        # print("Data Width:", dataIn.shape[1])

        outHeight = int((inputHeight-kernelHeight) / verticalStrideSize)+1
        outWidth = int((inputWidth-kernelWidth) / horizontalStrideSize)+1

        # print("Out Height:", outHeight)
        # print("Out Width:", outWidth)

        verticalStrideCount = int(outHeight/verticalStrideSize)
        horizontalStrideCount = int(outWidth/horizontalStrideSize)

        # print("Vertical stride count:", verticalStrideCount)
        # print("Horizontal stride count:", horizontalStrideCount)

        outputArray = np.zeros((outHeight, outWidth))

        for i in range(verticalStrideCount):
            for j in range(horizontalStrideCount):
                # print(i,j)
                x = dataIn[i:i+kernelWidth, j:j+kernelWidth]
                # print(x)
                outputArray[i, j] = np.sum(x * self.kernel)

        return outputArray


    def backwardPropagate(self, gradIn):
        pass