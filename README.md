# Evaluating Model Perception of Color Illusions in Photorealistic Scenes

Authors: Lingjun Mao, Zineng Tang, Alane Suhr

---

![examples](https://github.com/mao1207/RCID/blob/main/images/example.gif?raw=true)


## Abstract

We study the perception of color illusions by vision-language models. **Color illusion**, where a person's visual system perceives color differently from actual color, is well-studied in human vision. However, it remains underexplored whether vision-language models (VLMs), trained on large-scale human data, exhibit similar perceptual biases when confronted with such color illusions. We propose an automated framework for generating color illusion images, resulting in **RCID** (Realistic Color Illusion Dataset), a dataset of 19,000 realistic illusion images. Our experiments show that all studied VLMs exhibit perceptual biases similar to human vision. Finally, we train a model to distinguish both human perception and actual pixel differences.

## Contributions

1. We propose an automated framework for generating realistic illusion images and create a large, realistic dataset of color illusion images, named **Realistic Color Illusion Dataset (RCID)**, to enhance the fairness and accuracy of model testing.

2. We investigate underlying mechanisms of color illusions in VLMs, highlighting the combined influence of the vision system and prior knowledge. We also explore how external prompts and instruction tuning impact the models' performance on these illusions.

3. We propose a simple training method that enables models to understand human perception while also recognizing the actual pixel values.


## Dataset

![RCID](https://github.com/mao1207/RCID/blob/main/images/main_figure.png?raw=true)

The construction of our dataset involves three steps:

1. **Image Generation.** For contrast and stripe illusions, we use procedural code to generate simple illusion images, which are then processed by ControlNet to create realistic illusion images. For filter illusions, we directly apply contrasting color filters to the original images. Each type of illusion also includes a corresponding control group without any illusions for comparison.

2. **Question Generation.** We use GPT-4o to generate image-specific questions that are designed to evaluate the model's understanding of the illusion.

3. **Human Feedback.** We collect human participants' feedback on these images and adjust the original classification of “illusion” and “non-illusion” based on whether participants are deceived.

Our data can be found in the following link: [RCID Dataset](https://huggingface.co/datasets/mao1207/RCID)

The model used to generate the data is available at:  [Color Diffusion](https://huggingface.co/mao1207/color-diffusion).

## License

The source code of this repository is released under the Apache License 2.0. The model license and dataset license are listed on their corresponding webpages.

For more information, access to the dataset, and to contribute, please visit our [Website](https://color-illusion.github.io/).
