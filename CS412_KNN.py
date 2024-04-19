import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Testimg Data To Use
poker_test = np.loadtxt("poker-hand-testing.data", dtype='str')
#Training Data To Use
poker_train = np.loadtxt("poker-hand-training-true.data", dtype='str')

# Convert string data to integers
def convert_to_int(data):
    converted_data = []
    for line in data:
        converted_line = [int(x) for x in line.split(',')]
        converted_data.append(converted_line)
    return np.array(converted_data)

poker_test = convert_to_int(poker_test)
poker_train = convert_to_int(poker_train)

# Split features and labels
X_train, y_train = poker_train[:, :-1], poker_train[:, -1]
X_test, y_test = poker_test[:, :-1], poker_test[:, -1]

# Print the shapes of the training and testing data
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

#KNN algorithm
def knn_predict(X_train, y_train, x_test, k):
    distances = np.sum((X_train - x_test)**2, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_neighbors = X_train[nearest_indices]
    nearest_labels = y_train[nearest_indices]
    unique_classes, class_counts = np.unique(nearest_labels, return_counts=True)
    predicted_class = unique_classes[np.argmax(class_counts)]
    return predicted_class, nearest_neighbors, nearest_labels

# test instance
x_test = X_test[0].reshape(1, -1)

# Predicting class using KNN
predicted_class, nearest_neighbors, nearest_labels = knn_predict(X_train, y_train, x_test, k=5)
print("Predicted class:", predicted_class)

plt.figure(figsize=(10, 6))

# Plot each class with different colors
for i, (label, color) in enumerate(zip(range(10), ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta'])):
    plt.scatter(X_train[y_train == label][:, 0], X_train[y_train == label][:, 1], color=color, label=f'Class {label}', marker='o', alpha=0.5)  # Set alpha to 0.5 for transparency

# Plot testing sample
plt.scatter(x_test[:, 0], x_test[:, 1], color='black', marker='^', label='Testing Sample', alpha=0.5)  # Set alpha to 0.5 for transparency

# Plot circles around K Nearest Neighbors
for neighbor in nearest_neighbors:
    circle = plt.Circle((neighbor[0], neighbor[1]), 0.25, color='green', fill=False, alpha=0.5)  # Set alpha to 0.5 for transparency
    plt.gca().add_artist(circle)

# Adjust plot limits
plt.xlim(0, 5)
plt.ylim(0, 14)

plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classification with Nearest Neighbors')
plt.grid(True)
plt.show()
