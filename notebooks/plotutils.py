from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

from wholeslidedata.samplers.utils import fit_data
from wholeslidedata.samplers.utils import plot_mask


def plot_mfmr_patches(target_patch, context_patch, size, downsampling):
    downsampling_size = size / downsampling
    downsampling_coordinate =  size//2 - (downsampling_size//2)
    rect = patches.Rectangle((downsampling_coordinate, 
                              downsampling_coordinate), 
                             downsampling_size, 
                             downsampling_size, 
                             linewidth=3, 
                             edgecolor='lightgreen', 
                             facecolor='none')

    rect2 = patches.Rectangle((1,1),size-2,size-2, 
                             linewidth=4, 
                             edgecolor='lightgreen', 
                             facecolor='none')

    fig, axes = plt.subplots(1,2, figsize=(10,10))
    axes[0].imshow(target_patch)
    axes[0].add_patch(rect2)
    axes[1].imshow(context_patch)
    axes[1].add_patch(rect)
    plt.show()

def plot_sample(x_batch, y_batch):
    
    fig, axes = plt.subplots(2, 5, figsize=(10,5))

    axes[0][0].imshow(x_batch[0])
    axes[0][1].imshow(fit_data(x_batch[0], [70,70]))
    axes[0][2].set_title(str(np.unique(y_batch[0][..., 0])))
    axes[0][2].imshow(y_batch[0][..., 0])

    axes[0][3].set_title(str(np.unique(y_batch[0][..., 1])))
    axes[0][3].imshow(y_batch[0][..., 1])

    axes[0][4].set_title(str(np.unique(y_batch[0][..., 2])))
    axes[0][4].imshow(y_batch[0][..., 2])

    axes[1][0].imshow(x_batch[1])
    axes[1][1].imshow(fit_data(x_batch[1], [70,70]))
    axes[1][2].set_title(str(np.unique(y_batch[1][..., 0])))
    axes[1][2].imshow(y_batch[1][..., 0])
    axes[1][3].set_title(str(np.unique(y_batch[1][..., 1])))
    axes[1][3].imshow(y_batch[1][..., 1])
    axes[1][4].set_title(str(np.unique(y_batch[1][..., 2])))
    axes[1][4].imshow(y_batch[1][..., 2])

    plt.tight_layout()
    plt.show()
    
    
def plot_inference(patch, ground_truth, prediction):
    colors = ['black', 'red', 'pink', 'purple']
    colors = ['black', 'red', 'pink', 'purple']
    fig, axes = plt.subplots(1,3, figsize=(10,10))
    axes[0].imshow(patch)
    plot_mask(ground_truth, axes=axes[1], color_values=colors)
    plot_mask(prediction, axes=axes[2], color_values=colors)
    plt.show()