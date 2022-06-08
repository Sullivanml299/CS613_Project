import numpy as np
from PIL import Image
from ImageEnhancer import ImageEnhancer
from  matplotlib import  pyplot as plt

names = ["Output1000.png",
         "Output2000.png",
         "Output3000.png",
         "Output4000.png",
         "Output5000.png",
         "Output6000.png",
         "Output7000.png",
         "Output8000.png",
         "Output9000.png",
         "Output10000.png"]

enhanced = ["enhanced_Output1000.png",
            "enhanced_Output2000.png",
            "enhanced_Output3000.png",
            "enhanced_Output4000.png",
            "enhanced_Output5000.png",
            "enhanced_Output6000.png",
            "enhanced_Output7000.png",
            "enhanced_Output8000.png",
            "enhanced_Output9000.png",
            "enhanced_Output10000.png"]


ie = ImageEnhancer()

for name in names:
    sprite = np.array(Image.open(name))
    s2 = ie.enhanceScaled(arr=sprite)
    s2 = Image.fromarray(s2)
    s2.show()
    s2.save("enhanced_"+name)


fig = plt.figure(figsize=(10, 2))

for i in range(len(names)):
    image = Image.open(names[i])
    plt.subplot(2, 10, i+1)
    plt.imshow(image)
    plt.title((i+1)*1000, fontsize=10)
    plt.axis('off')


for i in range(len(enhanced)):
    image = Image.open(enhanced[i])
    plt.subplot(2, 10, i+11)
    plt.imshow(image)
    plt.axis('off')

plt.show()