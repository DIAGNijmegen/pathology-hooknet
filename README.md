###  HookNet - multi-resolution convolutional neural networks for semantic segmentation

----

#### Dependencies
This code has been tested on Ubuntu 18.04, tensorflow-gpu==2.3.0
 
#### Examples
Please see  [HookNet - practical guide](https://github.com/DIAGNijmegen/pathology-hooknet/blob/master/notebooks/HookNetPracticalGuide.ipynb) for an example on how to train/apply HookNet

#### Additional Information


###### Paper
This model is presented in our paper: 

[HookNet: Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images](https://www.sciencedirect.com/science/article/pii/S1361841520302541)

If you use this code, please cite the paper:

```
@article{VANRIJTHOVEN2021101890,
title = {HookNet: Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images},
journal = {Medical Image Analysis},
volume = {68},
pages = {101890},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2020.101890},
url = {https://www.sciencedirect.com/science/article/pii/S1361841520302541},
author = {Mart {van Rijthoven} and Maschenka Balkenhol and Karina Siliņa and Jeroen {van der Laak} and Francesco Ciompi},
keywords = {Computational pathology, Semantic segmentation, Multi-resolution, Deep learning},
abstract = {We propose HookNet, a semantic segmentation model for histopathology whole-slide images, which combines context and details via multiple branches of encoder-decoder convolutional neural networks. Concentric patches at multiple resolutions with different fields of view, feed different branches of HookNet, and intermediate representations are combined via a hooking mechanism. We describe a framework to design and train HookNet for achieving high-resolution semantic segmentation and introduce constraints to guarantee pixel-wise alignment in feature maps during hooking. We show the advantages of using HookNet in two histopathology image segmentation tasks where tissue type prediction accuracy strongly depends on contextual information, namely (1) multi-class tissue segmentation in breast cancer and, (2) segmentation of tertiary lymphoid structures and germinal centers in lung cancer. We show the superiority of HookNet when compared with single-resolution U-Net models working at different resolutions as well as with a recently published multi-resolution model for histopathology image segmentation. We have made HookNet publicly available by releasing the source code11https://github.com/computationalpathologygroup/hooknet as well as in the form of web-based applications22https://grand-challenge.org/algorithms/hooknet-breast/.,33https://grand-challenge.org/algorithms/hooknet-lung/. based on the grand-challenge.org platform.}
}
```


###### Pre-trained models

A pretraind model on breast or lung can be applied via https://grand-challenge.org/. Please create an user account and request access to an algorithm if you are interested. 

You can try out a pretrained HookNet on breast tissue here:  
https://grand-challenge.org/algorithms/hooknet/  

You can try out a pretrained HookNet on lung tissue here:  
https://grand-challenge.org/algorithms/hooknet-lung/

