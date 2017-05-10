import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np

__all__ = ['imshow', 'show_batch', 'show_mask']

# functions to show an image
def imshow(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = imshow(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()


def show_mask(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    plt.figure(1)
    images = imshow(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(211)
    plt.imshow(images)

    masks = imshow(torchvision.utils.make_grid(mask))
    plt.subplot(212)
    plt.imshow(masks)

    plt.show()