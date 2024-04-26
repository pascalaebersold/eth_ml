# utils imports
from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results

# time imports
from time import time

# import open cv
import cv2

# scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt

# numpy imports
import numpy as np

# scikit-image import
from skimage import data, img_as_float
from skimage import exposure

def extract_features(image_paths):
    image = cv2.imread(image_paths)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist



if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    image_paths = ["/data/train_images/"]
    #X = np.array(images)
    y = np.array(distances)

    X = [extract_features(image_path) for image_path in image_paths]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


    model = RandomForestClassifier(n_estimators=100, n_jobs=8)
    start = time()
    model.fit(X_train, y_train)
    stop = time()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("[Accuracy]:", acc)
    print("[Time]:", stop-start)

    # TODO: Your implementation starts here

    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(private_test_pred)
