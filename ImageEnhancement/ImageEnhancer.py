import numpy as np


class ImageEnhancer:
    def enhanceFlat(self, arr, threshold=5, amount=50):
        # arr should be a 3 channel array
        channels = arr.shape[2]
        for c in range(channels):
            _c = arr[:, :, c]
            print(_c.shape)
            max = np.max(_c)
            print(max)
            _c[_c > threshold] += amount

        return arr

    def enhanceScaled(self, arr, threshold=5, amount=15):
        # arr should be a 3 channel array
        channels = arr.shape[2]
        for c in range(channels):
            _c = arr[:, :, c]
            _c[_c > threshold] *= amount

        return arr