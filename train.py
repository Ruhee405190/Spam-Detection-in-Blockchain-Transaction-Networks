import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = "dataset.csv"  # Update with your dataset's file path
df = pd.read_csv(file_path)

# Clean up column names by stripping extra spaces
df.columns = df.columns.str.strip()

# Define target and features
target_column = "FLAG"
features = [col for col in df.columns if col != target_column]

# Drop non-numeric columns, including 'Address'
df = df.select_dtypes(include=[float, int])

# Explicitly remove 'Address' from the feature list if it still exists
features = [col for col in features if col != 'Address']

# Check for missing columns in the feature list
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    print(f"Warning: The following columns are missing and will be excluded from the feature list: {missing_columns}")
    features = [col for col in features if col in df.columns]

# Split data into features (X) and target (y)
X = df[features]
y = df[target_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
model_file = "flag_prediction_gb_model2.pkl"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

# Plot meaningful charts

# 1. Distribution of the target variable (FLAG)
plt.figure(figsize=(8, 6))
sns.countplot(x='FLAG', data=df)
plt.title("Distribution of FLAG (Target Variable)")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 3. Feature Importance from Gradient Boosting
importances = model.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importance from Gradient Boosting")
plt.xlabel("Importance")
plt.show()

# 4. Boxplot of Ether Sent vs Ether Received
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['total Ether sent', 'total ether received']])
plt.title("Boxplot of Ether Sent vs Ether Received")
plt.show()

# 5. Sent Transactions vs Received Transactions (Scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sent tnx', y='Received Tnx', hue='FLAG', data=df, palette="Set1")
plt.title("Sent Transactions vs Received Transactions")
plt.show()


# 7. Time Difference Between First and Last Transactions (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(df['Time Diff between first and last (Mins)'], bins=30, kde=True)
plt.title("Time Difference Between First and Last Transactions")
plt.show()

# 8. Unique Sent vs Unique Received Addresses (Bar chart)
plt.figure(figsize=(8, 6))
sns.barplot(x='Unique Sent To Addresses', y='Unique Received From Addresses', data=df)
plt.title("Unique Sent vs Unique Received Addresses")
plt.show()


# 10. Average Time Between Sent and Received Transactions
plt.figure(figsize=(8, 6))
sns.lineplot(x='Avg min between sent tnx', y='Avg min between received tnx', data=df, marker="o", color='green')
plt.title("Average Time Between Sent and Received Transactions")
plt.show()

