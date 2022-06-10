import numpy as np


def deleteNoiseByNeighbors(img, Neighbor = [-1,0,1], minNeighborAmount = 3):
    
    hight, width  = img.shape
    outImg = np.zeros((hight, width))
    for i in range(hight):
        for j in range(width):
            if img[i,j] == 0:
                continue
            counter = 0
            for k in Neighbor:
                for l in Neighbor:
                    try:
                        if img[i+k,j+l] == 255:
                            counter+=1
                    except:
                        pass
            # -1 ==> don't count a pixel as it self's neighbor!
            if counter -1 > minNeighborAmount:
                outImg[i,j] = 1

    return outImg


def deleteNoneBinaryPixels(img):
    half = 255/2
    hight, width = img.shape
    
    for i in range(hight):
        for j in range(width):
            if img[i,j] > half:
                img[i,j] = 255
            else:
                img[i,j] = 0

    return img