import tensorflow as tf
import cv2

# Load the model from the .h5 file
model = tf.keras.models.load_model('C:/Users/emana/OneDrive/Desktop/Face Mask Detection/face-mask-try-1/face_mask_detection_model.h5')

# Define a list of class labels
class_labels = ['mask', 'no_mask']

# Load the input image
image = cv2.imread('C:/Users/emana/OneDrive/Desktop/Face Mask Detection/face-mask-try-1/dataset/test\with_mask/0_0_0 copy 46.jpg')

# Preprocess the input image
image_orig = image.copy() # make a copy of the original image
image = cv2.resize(image, (200, 200)) # resize to the input size of the model
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB color space
image = image.astype('float32') / 255.0 # normalize pixel values to [0, 1]
image = tf.expand_dims(image, axis=0) # add batch dimension

# Use the model to make predictions on the input image
predictions = model.predict(image)

# Get the predicted class label
predicted_class_index = tf.argmax(predictions, axis=-1).numpy()[0]
predicted_class_label = class_labels[predicted_class_index]

# Display the input image and predicted class label
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image_orig, predicted_class_label, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Image with Predicted Label', image_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()