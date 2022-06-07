from math import floor, ceil
import numpy as np

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
        print("Output: \n {}".format(output))

        # Calculate the output
        # TODO: see if we can leverate linear algebra here
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
            print("Output: \n {}".format(output))

        # Define length of padding
        p_left, p_right, p_top, p_bottom = self.setPadding()

        # Add padding
        output_padded = output[p_left:output.shape[0]-p_right, p_top:output.shape[0]-p_bottom]

        return(np.array(output_padded))

    def setPadding(self):
        if self.padding == "same":
            # returns the output with the shape of (input shape)*(stride)
            p_left = floor((self.W.shape[0] - self.strides[0])/2)
            p_right = self.W.shape[0] - self.strides[0] - p_left
            p_top = floor((self.W.shape[1] - self.strides[1])/2)
            p_bottom = self.W.shape[1] - self.strides[1] - p_left
        elif self.padding == "valid":
            # returns the output without any padding
            p_left = 0
            p_right = 0
            p_top = 0
            p_bottom = 0
        else:
            return 0, 0, 0, 0

        return p_left, p_right, p_top, p_bottom


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

        print("Kernel Height:", kernelHeight)
        print("Kernel Width:", kernelWidth)

        print("Data Height:", dataIn.shape[0])
        print("Data Width:", dataIn.shape[1])

        outHeight = int((inputHeight-kernelHeight) / verticalStrideSize)+1
        outWidth = int((inputWidth-kernelWidth) / horizontalStrideSize)+1

        print("Out Height:", outHeight)
        print("Out Width:", outWidth)

        verticalStrideCount = int(outHeight/verticalStrideSize)
        horizontalStrideCount = int(outWidth/horizontalStrideSize)

        print("Vertical stride count:", verticalStrideCount)
        print("Horizontal stride count:", horizontalStrideCount)

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