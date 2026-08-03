import random

class SimpleClassifier:
    def __init__(self, majority_class_prediction_rate=0.994):
        # This classifier is designed to simulate a scenario where it always predicts the majority class.
        # The 'majority_class_prediction_rate' represents the accuracy achieved by this simple strategy.
        self.majority_class_prediction_rate = majority_class_prediction_rate
        self.majority_class = 0 # Assume class 0 is the majority class
        self.minority_class = 1

    def predict(self, data_point):
        # Always predict the majority class, regardless of the input data.
        # This is the core of the misleading accuracy demonstration.
        return self.majority_class

    def evaluate(self, dataset):
        correct_predictions = 0
        total_samples = len(dataset)

        for actual_class, _ in dataset:
            predicted_class = self.predict(None) # Input data is ignored by this classifier
            if predicted_class == actual_class:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        return accuracy

def generate_imbalanced_dataset(size=1000, minority_ratio=0.006):
    dataset = []
    # Generate majority class samples
    num_majority = int(size * (1 - minority_ratio))
    for _ in range(num_majority):
        dataset.append((0, 'feature_data_majority')) # (actual_class, data)

    # Generate minority class samples
    num_minority = size - num_majority
    for _ in range(num_minority):
        dataset.append((1, 'feature_data_minority')) # (actual_class, data)

    random.shuffle(dataset)
    return dataset

# --- Demonstration ---

# 1. Create an imbalanced dataset
# Let's simulate a scenario where class 1 is very rare (e.g., a rare disease)
# The minority_ratio is set to 0.006, meaning only 0.6% of the data belongs to class 1.
# This means class 0 (majority class) makes up 99.4% of the data.
imbalanced_data = generate_imbalanced_dataset(size=1000, minority_ratio=0.006)

# 2. Instantiate a classifier that *always* predicts the majority class (class 0)
# This classifier will achieve an accuracy of 99.4% simply by always guessing the most frequent class.
# This is the core of the misleading accuracy.
simple_majority_classifier = SimpleClassifier(majority_class_prediction_rate=0.994)

# 3. Evaluate the classifier's accuracy on the imbalanced dataset
calculated_accuracy = simple_majority_classifier.evaluate(imbalanced_data)

print(f"Dataset size: {len(imbalanced_data)}")
print(f"Majority class (0) proportion: {1 - 0.006:.3f}")
print(f"Minority class (1) proportion: {0.006:.3f}")
print(f"Classifier always predicts majority class (0).")
print(f"Calculated Accuracy: {calculated_accuracy:.3f}")

# Analyze the results:
# The accuracy is 99.4%, which sounds excellent. However, this classifier completely fails to detect
# any instances of the minority class (class 1). In real-world scenarios like fraud detection or
# disease diagnosis, failing to detect the rare but critical event is unacceptable, even with high accuracy.

print("\n--- Interpretation ---")
print("While the accuracy is 99.4%, this classifier is useless for detecting the minority class (class 1).")
print("This highlights why accuracy alone can be misleading on imbalanced datasets.")
print("Other metrics like Precision, Recall, F1-score, or AUC are often more informative.")
