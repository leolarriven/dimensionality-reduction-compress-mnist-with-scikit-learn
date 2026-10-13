"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist_subset
import numpy as np
import pandas as pd
import urllib.request
import os
import tempfile

def load_mnist_subset(n=3000):
    # TODO: cache mnist.npz in tempfile.gettempdir(); return (X (n, 784) float64, y int64).
    path = os.path.join(tempfile.gettempdir(), 'mnist.npz')
    if not os.path.exists(path):
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', path)
    data = np.load(f'{path}')


    X = data['x_train']
    X = X[:n].reshape(n, -1).astype('float64')
    Y = data['y_train'][:n].astype('int64')

    return X, Y

# Step 2 - fit_pca (not yet solved)
# TODO: implement

# Step 3 - components_for_variance (not yet solved)
# TODO: implement

# Step 4 - explained_variance_curve (not yet solved)
# TODO: implement

# Step 5 - compress (not yet solved)
# TODO: implement

# Step 6 - reconstruction_error (not yet solved)
# TODO: implement

# Step 7 - compression_ratio (not yet solved)
# TODO: implement

# Step 8 - randomized_pca (not yet solved)
# TODO: implement

# Step 9 - incremental_pca (not yet solved)
# TODO: implement

# Step 10 - random_projection (not yet solved)
# TODO: implement

# Step 11 - unroll_swiss_roll (not yet solved)
# TODO: implement

# Step 12 - tsne_map (not yet solved)
# TODO: implement

# Step 13 - classifier_on_compressed (not yet solved)
# TODO: implement

# Step 14 - pca_classifier_pipeline (not yet solved)
# TODO: implement

# Step 15 - save_and_reload_pipeline (not yet solved)
# TODO: implement

# Step 16 - predict_images (not yet solved)
# TODO: implement

