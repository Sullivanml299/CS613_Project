from PIL import Image
import numpy as np
from scipy import linalg



def FID(X, Y):
    X = X[:,:,0] + X[:,:,1] + X[:,:,2]
    Y = Y[:,:,0] + Y[:,:,1] + Y[:,:,2]
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    print("mu_X:", mu_x, "mu_Y:", mu_y)

    ssdiff = np.sum((mu_x - mu_y)**2.0)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    covmean = linalg.sqrtm(cov_x.dot(cov_y))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score

    fid = ssdiff + np.trace(cov_x + cov_y - 2.0 * covmean)
    return fid

realSprite = np.array(Image.open("real.png"))
generatedSprite_1 = np.array(Image.open("Output1000.png"))
generatedSprite_enhanced_1 = np.array(Image.open("enhanced_Output1000.png"))
generatedSprite_10 = np.array(Image.open("Output10000.png"))
generatedSprite_enhanced_10 = np.array(Image.open("enhanced_Output10000.png"))

# print("GENERATED")
#
# print("ENHANCED")

f1 = FID(realSprite, generatedSprite_1)
f1e = FID(realSprite, generatedSprite_enhanced_1)

f10 = FID(realSprite, generatedSprite_10)
f10e = FID(realSprite, generatedSprite_enhanced_10)

print(f1, f1e, f10, f10e)

max = np.max([f1, f1e, f10, f10e])

f1 /= max
f1e /= max
f10 /= max
f10e /= max


print(f1, f1e, f10, f10e)