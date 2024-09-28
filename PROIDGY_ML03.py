import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Set the directory paths
cat_dir = 'E:\\Downloads\\Pet_images\\cats'
dog_dir = 'E:\\Downloads\\Pet_images\\dogs'

# Function to load images and resize them
def load_images(directory, label, img_size=(64, 64)):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels


# Load cat and dog images
cat_images, cat_labels = load_images(cat_dir, 'cat')
dog_images, dog_labels = load_images(dog_dir, 'dog')


# Combine data and labels
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

# Flatten the images (convert to feature vectors)
images_flattened = images.reshape(images.shape[0], -1)

# Encode labels (convert 'cat' and 'dog' to 0 and 1)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels_encoded, test_size=0.2, random_state=42)

# Create the SVM model
svm = SVC(kernel='linear') 

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Optional: Predict on a new image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64,64))    
    img_flattened = img.flatten().reshape(1, -1)
    prediction = svm.predict(img_flattened)
    label = le.inverse_transform(prediction)
    return label[0]


# Image visulizing
image_path = 'E:\\Downloads\\Pet_images\\dogs\\dog_173.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: The image at {image_path} could not be loaded.")
else:
    # If the image loaded correctly, proceed with resizing
    resized_image = cv2.resize(image, (128, 128))  # Resize to 128x128 as an example
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 # Example prediction
example_image = 'E:\\Downloads\\Pet_images\\dogs\\dog_173.jpg'
prediction = predict_image(example_image)
print(f"The predicted label for the image is: {prediction}")   