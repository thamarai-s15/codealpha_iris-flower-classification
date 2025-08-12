# codealpha_iris-flower-classification
Iris Flower Classification ● Use measurements of Iris flowers (setosa, versicolor, virginica) as input data. ● Train a machine learning model to classify the species based on these measurements. ● Use libraries like Scikit-learn for easy dataset access and model building. ● Evaluate the model’s accuracy and performance using test data. 
# Iris Flower Classification with Machine Learning

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the Iris Dataset
iris = datasets.load_iris()

# Convert to DataFrame for easy handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target numbers to species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of Iris Dataset:")
print(df.head())

# 3. Features & Target
X = iris.data
y = iris.target

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 8. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
