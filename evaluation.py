import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from model import load_model  # Import the load_model function

# Load your model
model_path = 'C:\\SignLanguage\\action.h5'  # Update this path
model = load_model(model_path)

print(model.input_shape)

# Load your test data
X_test = np.load('X_test.npy')  # Load your test data
y_test = np.load('y_test.npy')  # Load your true labels

# Perform predictions
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# Evaluate the model
confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

print(confusion_matrix)
print(f'Accuracy: {accuracy}')
