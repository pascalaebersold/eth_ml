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
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score


from sklearn.pipeline import Pipeline


from sklearn.svm import NuSVR
from sklearn.decomposition import TruncatedSVD

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

    
    #scaler = StandardScaler()
    #scaler.fit(X_train)

    
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #svr_model = NuSVR(nu = 0.6, C = 3, kernel = 'rbf', degree = 3, gamma = 0.00055, coef0 = 0.0, shrinking = True, cache_size = 2000, tol = 0.001, verbose = False, max_iter = -1)
    #svr_model.fit(X_train_scaled, y_train)

    #pipeline.fit(X_train,y_train)
    #('pca', PCA(n_components = 100, svd_solver='full')),

    

    pipe = Pipeline([
        ('scaler', RobustScaler(with_centering = True, with_scaling = True, quantile_range = (30.0,70.0), copy = True)), 
        ('svd', TruncatedSVD(n_components = 90, algorithm = 'randomized', n_oversamples = 12, power_iteration_normalizer = 'none')),
        ('NuSVR',NuSVR(nu = 0.6, C = 3, kernel = 'rbf', degree = 2, gamma = 0.00055, coef0 = 0.0, shrinking = True, cache_size = 2000, tol = 0.0001, verbose = False, max_iter = -1))
        ])
    
    #perform cross-validation
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')
    print("Cross_validation MSE:", -scores.mean())

    pipe.fit(X_train,y_train)

    #y_pred_train = svr_model.predict(X_train_scaled)
    #y_pred = svr_model.predict(X_test_scaled)
    y_pred_train = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)
    
    

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
    print_results(y_test,y_pred)

    # compare predicted distance to actual distance
    """for i in range(len(X_test_scaled)):
        pred = svr_model.predict(np.array([X_test_scaled[i]]))
        expected = y_test[i]
        print(f"actual distance: {expected}, predicted distance: {pred}")
        input()"""

    # load private test images and guess for yourself
    '''private_test_images = load_private_test_dataset(config)
    for image in private_test_images:
        img = scaler.transform(np.array([image]))
        pred = svr_model.predict(img)
        print(f"predicted distance: {pred}")
        print(img.shape)
        image = image.reshape((300//config["downsample_factor"], 300//config["downsample_factor"], 3))
        plt.imshow(image)
        plt.show()
        input()'''