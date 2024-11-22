import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from ece import expected_calibration_error

x_train = np.load('../data/differences_train.npy')
x_test = np.load('../data/differences_test.npy')
y_train = np.load('../data/labels_train.npy')
y_test = np.load('../data/labels_test.npy')

kernel = 1.0 * RBF(length_scale=1.0)
gp_classifier = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
gp_classifier.fit(x_train, y_train)

y_pred_probs = gp_classifier.predict_proba(x_test)[:, 1]
y_pred_labels = (y_pred_probs > 0.5)

train_accuracy = accuracy_score(y_train, gp_classifier.predict(x_train))
print(f'Train Accuracy: {train_accuracy:.4f}')

test_accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Test Accuracy: {test_accuracy:.4f}')

expected_calibration_error(y_pred_probs, y_test, model_name="Gaussian Process Classifier")
