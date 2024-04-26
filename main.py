# utils imports
from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results

# scikit-learn imports
from sklearn import linear_model

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt

# numpy imports
import numpy as np

# scikit-image import
from skimage import data, img_as_float
from skimage import exposure



if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # TODO: Your implementation starts here

    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(private_test_pred)