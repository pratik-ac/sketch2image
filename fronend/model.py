import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

# Load model
model = load_model('model.h5', custom_objects={'dice_loss': dice_loss})

# Load and preprocess image
img = cv2.imread('image.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#img = cv2.resize(img, (178, 218))  # Resize to match model input size
#img = img.astype(np.float32) / 255.0  # Normalize the image to range [0, 1]

# Visualize the original and preprocessed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread('image.jpg'), cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title('Preprocessed Image')

plt.show()

# Predict with the model (reshape image for model input)
#img_input = img.reshape(1, 218, 178, 3)
y_pred = model.predict(img_input)

# Visualize the predicted image
plt.figure(figsize=(10, 5))
plt.imshow(y_pred[0])  # Display the predicted image
plt.title('Predicted Image')
plt.show()

# Optionally, save the predicted image
#matplotlib.image.imsave('predicted_image.png', y_pred[0])

# Check max value of the image for diagnostics
print(f"Max value in the image: {np.max(img.flatten())}")
