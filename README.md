## Dog segmentation

Just a simple project discovering latest segmentation trends.

1. Grab photos of dogs (publicly available) from LAION-5B dataset (using [this](https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn5.laion.ai&index=laion5B&useMclip=false) tool)
2. Filtered and hand-labeled subset of photos using interactive segmentation tool [focalclick](https://github.com/XavierCHEN34/ClickSEG)
3. using this pipeline trained big model for semantic segmentation
4. using active learning labeled more images and re-trained model
5. using knowledge distillation on [EfficientViT](https://arxiv.org/abs/2205.14756) model