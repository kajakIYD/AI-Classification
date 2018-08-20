import matplotlib.pyplot as plt
import numpy as np

def test():
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()

def showImg(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.gca().grid(False)
    plt.show()

def showImages(train_images, train_labels, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def showAndCompareInvalidPredictions(invalidPredictions, test_images, class_names):
    plt.figure(figsize=(10,10))
    for i in range(0, len(invalidPredictions)-1):
        plt.subplot(len(invalidPredictions), 1, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[invalidPredictions[i][1]] + ' should be: ' + class_names[invalidPredictions[i][2]])
    plt.show()


def filterTestImages(test_images, invalidPredictions):
    filteredImages = []
    j = 0
    for i in range(0, len(test_images)):
        if (i == invalidPredictions[j][0]):
            filteredImages.append(test_images[i])
            j +=1
            if (j==len(invalidPredictions)):
                break
    return filteredImages


def main():
    print("python main function")


if __name__ == '__main__':
    main()