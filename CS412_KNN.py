import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

#Testing Data To Use
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

# #KNN algorithm
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

#CROSS VALIDATION

# Define values of k to be tested
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
num_folds = 5
avg_accuracies = np.zeros(len(k_values))

# Perform cross-validation
for i, k in enumerate(k_values):
    fold_accuracies = []
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_fold, y_train_fold)

        # Predict labels for validation set
        y_val_pred = knn.predict(X_val_fold)

        # Calculate accuracy
        fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(fold_accuracy)

    avg_accuracy = np.mean(fold_accuracies)
    avg_accuracies[i] = avg_accuracy

    print(f"K = {k}:")
    for fold, accuracy in enumerate(fold_accuracies):
        print(f" Fold {fold+1} Accuracy: {accuracy}")
    print(f" Average Accuracy: {avg_accuracy}\n")

# Find optimal k
optimal_k_idx = np.argmax(avg_accuracies)
optimal_k = k_values[optimal_k_idx]
optimal_accuracy = avg_accuracies[optimal_k_idx]
print(f"Optimal K: {optimal_k} (Average Accuracy: {optimal_accuracy})")


# # Average accuracies
avg_accuracies = [
    0.5128, 0.5360, 0.5500, 0.5591, 0.5677, 
    0.5705, 0.5728, 0.5776, 0.5784, 0.5776
]

# Optimal k
optimal_k = 17

# Plot average accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, avg_accuracies, marker='o', color='b', linestyle='-')
plt.scatter(optimal_k, avg_accuracies[k_values.index(optimal_k)], color='r', s=100, label=f'Optimal K ({optimal_k})')
plt.title('Average Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Average Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# Train KNN classifier with optimal k
k_optimal = 17
knn_classifier = KNeighborsClassifier(n_neighbors=k_optimal)
knn_classifier.fit(X_train, y_train)

# Predict labels for testing data
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on testing data:", accuracy)

#PLOTTING DISTRIBUTION

# Predict labels for the entire testing dataset
y_pred_test = y_pred

# Count the occurrences of each class in the actual and predicted labels
actual_counts = {label: (y_test == label).sum() for label in np.unique(y_test)}
predicted_counts = {label: (y_pred_test == label).sum() for label in np.unique(y_pred_test)}

# Ensure both dictionaries have counts for all classes
for label in np.unique(y_test):
    if label not in predicted_counts:
        predicted_counts[label] = 0
for label in np.unique(y_pred_test):
    if label not in actual_counts:
        actual_counts[label] = 0

# Plot the distribution of actual and predicted classes
plt.figure(figsize=(10, 6))
labels = np.arange(10)  # Assuming there are 10 classes
width = 0.35  # Bar width

plt.bar(labels - width/2, actual_counts.values(), width, color='blue', alpha=0.5, label='Actual')
plt.bar(labels + width/2, predicted_counts.values(), width, color='green', alpha=0.5, label='Predicted')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Distribution of Actual and Predicted Classes')
plt.xticks(labels)
plt.legend()
plt.show()


#LOG FREQUENCY
# Count the occurrences of each class in the actual and predicted labels
actual_counts = {label: (y_test == label).sum() for label in np.unique(y_test)}
predicted_counts = {label: (y_pred_test == label).sum() for label in np.unique(y_pred_test)}

# Ensure both dictionaries have counts for all classes
for label in np.unique(y_test):
    if label not in predicted_counts:
        predicted_counts[label] = 0
for label in np.unique(y_pred_test):
    if label not in actual_counts:
        actual_counts[label] = 0

# Plot the distribution of actual and predicted classes (using logarithm of counts)
plt.figure(figsize=(10, 6))
labels = np.arange(10)  # Assuming there are 10 classes
width = 0.35  # Bar width

plt.bar(labels - width/2, np.log(list(actual_counts.values())), width, color='blue', alpha=0.5, label='Actual')
plt.bar(labels + width/2, np.log(list(predicted_counts.values())), width, color='green', alpha=0.5, label='Predicted')
plt.xlabel('Class')
plt.ylabel('Log Frequency')
plt.title('Distribution of Actual and Predicted Classes (Log Scale)')
plt.xticks(labels)
plt.legend()
plt.show()
