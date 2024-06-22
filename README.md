# Geometric Shapes Image Similarity Model

This project aims to develop an image similarity model using neural networks to identify the similarity between geometric shapes.

## Dataset

The dataset used for this project can be found [here](https://data.mendeley.com/datasets/wzr2yv7r53/1). It consists of 9 geometric shapes (Triangle, Square, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Circle and Star), each of them drawn randomly on a 200x200 RGB image.

## Project Structure

- `train_and_eval.py`: Script for training and evaluating the model.
- `similarity.py`: Script for evaluating the similarity between two images.
- `helpers/`: Contains helper scripts for preprocessing and model definition.
  - `preprocessing.py`: Script for loading and preprocessing the data.
  - `model.py`: Script defining the model architecture.
- `dataset/`: Contains example images used for computing similarity between two shapes.
- `models/`: Directory where trained models are saved.
- `notebooks/`: Contains a Jupyter notebook that can be used to train the model on GPUs, e.g. using [Google Colab](https://colab.research.google.com/).

## Requirements

The following libraries are required:
- `torch`
- `torchvision`
- `scikit-learn`

they can be installed with:
```sh
pip install torch torchvision scikit-learn
```

## Usage

### Training 

Run the `train_and_eval.py` script to train and evaluate the model. This script will save the trained model in the `models/` directory.

```sh
python3 train_and_eval.py 
```
### Compute similarity 

Use the `similarity.py` script to evaluate the similarity between to shapes. 

```sh
python3 evaluate.py image1.png image2.png --model-path models/model.pth
```

