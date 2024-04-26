# utils imports
from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
from pathlib import Path

# scikit-learn imports
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt

# numpy imports
import numpy as np

# scikit-image import
# from skimage import data, img_as_float
from skimage import exposure

from PIL import Image


if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images_train, distances_train = load_dataset(config, split="train")
    print(f"[INFO]: Dataset loaded with {len(images_train)} samples.")
    images_test, distances_test = load_dataset(config, split="public_test")
    print(f"[INFO]: Dataset loaded with {len(images_test)} samples.")

    X_train, y_train = np.array(images_train), np.array(distances_train)
    X_test, y_test = np.array(images_test), np.array(distances_test)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr_model = SVR(kernel='rbf', C=2, cache_size=9000, shrinking=False, epsilon=0.2)
    svr_model.fit(X_train_scaled, y_train)

    y_pred_train = svr_model.predict(X_train_scaled)
    y_pred = svr_model.predict(X_test_scaled)

    # show mean squared error (not important)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (train): {mse_train}")
    print("Mean Squared Error (test): ", mse)

    # show maximum (not important) and mean difference
    diff_train = abs(y_train - y_pred_train)
    print(f"maximum difference (train): {diff_train.max()}, mean difference: {diff_train.mean()}")
    diff = abs(y_test - y_pred)
    print(f"maximum difference: {diff.max()}, mean difference: {diff.mean()}")

    # compare predicted distance to actual distance
    """for i in range(len(X_test_scaled)):
        pred = svr_model.predict(np.array([X_test_scaled[i]]))
        expected = y_test[i]
        print(f"actual distance: {expected}, predicted distance: {pred}")
        input()"""

    # load private test images and guess for yourself
    """private_test_images = load_private_test_dataset(config)
    for image in private_test_images:
        img = scaler.transform(np.array([image]))
        pred = svr_model.predict(img)
        print(f"predicted distance: {pred}")
        print(img.shape)
        image = image.reshape((30, 30, 3))
        plt.imshow(image)
        plt.show()
        input()"""