
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset from OpenML (70,000 samples of 28x28 images flattened to 784 features)
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data  # Feature set: pixel values
y = mnist.target.astype(int)  # Labels: digits 0-9

# Display an example digit
plt.imshow(X.iloc[0].values.reshape(28, 28), cmap='gray')
plt.title(f'Example Digit: {y.iloc[0]}')
plt.axis('off')
plt.show()

# Split dataset into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model with multinomial class setting
model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(X_train_scaled, y_train)
# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Test Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)
