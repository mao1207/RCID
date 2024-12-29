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

## Generate Your Dataset

### Conda Setup

#### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/mao1207/RCID.git
```

#### Step 2: Configure the Environment

After cloning the repository, set up your Conda environment. You need to install the following dependencies:

```bash
conda create -n rcid python=3.8
conda activate rcid
conda install pytorch torchvision -c pytorch
pip install accelerate>=0.16.0
pip install transformers>=4.25.1
pip install ftfy
pip install tensorboard
pip install datasets
```

Ensure that your environment is correctly configured with the above dependencies.

---

### Step-by-Step Guide to Dataset Generation

#### 1. Quantifying the Original Dataset

To simplify realistic images into basic shapes and colors for training ControlNet, first run the following Python script to quantize the original dataset images:

```bash
python ControlNet_training/generate_quantified_images.py
```

This script will convert realistic images into simplified versions to be used for training.

#### 2. Training ControlNet

Once the dataset is prepared, you can begin training ControlNet with the simplified dataset. Run the following command to start training:

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="Your_model_dir"

python train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/path/to/cocos-dataset/cocos \
 --resolution=512 \
 --learning_rate=5e-4 \
 --train_batch_size=4 \
 --num_train_epochs=20 \
 --checkpointing_steps=2000
```

- Replace `/path/to/cocos-dataset/cocos` with the actual path to your dataset.
- The `MODEL_DIR` should point to the pretrained model you want to use, for example, `runwayml/stable-diffusion-v1-5`.
- Adjust the `OUTPUT_DIR` to the location where you want to save the trained model.

#### 3. Generating Illusions

Once ControlNet has been trained, you can generate images with color illusions.

Run the following script to generate simplified images with color illusions:

```bash
python dataset_construction/simple_contrast_illusion_generation.py
```

This will create a set of simplified images with color illusions.

#### 4. Creating Realistic Illusions

To convert the simplified color illusion images into realistic images, use the following script:

```bash
python dataset_construction/realistic_illusion_generation.py
```

This script will generate realistic versions of the illusion images.

#### 5. Question Generation

Finally, generate corresponding questions for each image using the following command:

```bash
python dataset_construction/question_generation.py
```

This will generate questions related to the color differences and illusions present in the images.

---

### Directory Structure

Here’s a basic overview of the directory structure after running these steps:

```
RCID/
├── ControlNet_training/
│   ├── generate_quantified_images.py  # Quantize images for basic shapes and colors
│   └── train_controlnet.py           # Train the ControlNet model
├── dataset_construction/
│   ├── simple_contrast_illusion_generation.py  # Generate simplified contrast illusions
│   ├── simple_stripe_illusion_generation.py    # Generate simplified stripe illusions
│   ├── realistic_illusion_generation.py         # Generate realistic illusion images
│   ├── question_generation.py                  # Generate questions for datasets
│   └── filter_illusion_generation.py           # Generate filter illusion images
├── evaluation/
│   ├── eval                                  # Evaluation scripts
│   ├── model                                 # Pre-trained models
│   ├── serve                                 # Model serving code
│   └── train                                 # Training scripts for evaluation
├── dataset/                                  # Dataset directory
└── models/
    └── Your_trained_model/                   # Your trained ControlNet model
```

---

### Additional Notes

- Make sure your dataset is in the correct format before running the scripts.
- Ensure that you have sufficient disk space, as image generation and training require significant resources.
- You can adjust hyperparameters in the `train_controlnet.py` script based on your hardware and training requirements.

---

## License

The source code of this repository is released under the Apache License 2.0. The model license and dataset license are listed on their corresponding webpages.

For more information, access to the dataset, and to contribute, please visit our [Website](https://color-illusion.github.io/Color-Illusion/).
