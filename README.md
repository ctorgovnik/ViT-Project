# ViT-Project

## Introduction
Can the Transformer be applied effectively to images for large-scale recognition tasks? This question was addressed
by Dosovitskiy et al. from Google Research’s Brain Team in their 2020 paper, ”An Image is Worth 16×16
Words.”[2] The resulting Vision Transformer. (ViT) model achieved state-of-the-art results on several image classification benchmarks, demonstrating that attention based models could surpass convolutional architectures at scale.

## Chosen Result
We chose to reproduce the scaling results presented by Dosovitskiy et al. in their Vision Transformer (ViT)
paper[2]. Specifically, our goal was to replicate the findings that demonstrate how convolutional neural networks
(CNNs), particularly ResNets, outperform ViTs when trained on smaller datasets due to the strong inductive
biases inherent in convolutional architectures.

## GitHub Contents

## Re-implementation Details
As pointed out in the paper, training ViTs is very resource intensive. They are much more compute heavy then
convolutional models, which is one of their drawbacks. For us, as undergraduate students without access to
large compute, this meant we would have to find another way to test the scaling of these models. We decided
to mix models designed by ourselves with larger models pulled from resources on the internet.

## Reproduction Steps

### Using In House ViT

### Finetuning Pretrained Models From PyTorch
In [code\notebooks](code/notebooks) you will find the training loops for the models pulled from PyTorch. 

The file [ViTFinetuning.ipynb](code/notebooks/ViTFinetuning.ipynb) contains code to finetune the Base ViT and the Mini ViT as described in our report. Simply run the notebook from the top to finetune the models on CIFAR10 (done in Google Colab for compute and space to store datasets). To choose whether finetuning the Mini ViT or the Base ViT, simply change the flag ```UsingVIT_b``` between ```True``` and ```False```.

## Results/Insights
Our experiments reproduced the core finding of Dosovitskiy et al., showing that ViTs require large-scale pre-
training to outperform convolutional models like ResNet. On smaller datasets such as CIFAR-100, ResNet-18
showed stronger performance, but ViTs significantly outperformed when pretrained on ImageNet-1K. These
results highlight the scalability of ViTs and their potential when ample data and compute are available.

## Conclusion
In future work, we aim to investigate methods that reduce ViTs’ data and compute requirements, such
as knowledge distillation, data augmentation, or semi-supervised learning. We could also explore hybrid ar-
chitectures that combine convolutional and attention-based components to improve performance on smaller
datasets

## Acknowledgements
[1] Ashish Vaswani et al. “Attention is All You Need”. In: Advances in Neural Information Processing Systems.
Vol. 30. Curran Associates, Inc., 2017. <br><br>
[2] Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”.
In: International Conference on Learning Representations (ICLR). arXiv:2010.11929. 2021.<br><br>
[3] Alex Krizhevsky. Learning multiple layers of features from tiny images. Tech. rep. University of Toronto,
2009. url: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.<br><br>
[4] Facebook AI. deit-tiny-patch16-224 on HuggingFace. https://huggingface.co/facebook/deit-tiny-patch16-224. 2024.<br><br>
[5] PyTorch Team. Torchvision Models. https://pytorch.org/vision/main/models.html. 2024
