import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
import numpy as np

# Load the trained model
model_file = "flag_prediction_gb_model.pkl"
model = joblib.load(model_file)

input_data = {
    'Avg min between sent tnx': 27681.45,
    'Avg min between received tnx': 11171.03,
    'Time Diff between first and last (Mins)': 842599.05,
    'Sent tnx': 26,
    'Received Tnx': 11,
    'Unique Received From Addresses': 6,
    'Unique Sent To Addresses': 15,
    'total transactions (including tnx to create contract': 37,
    'total Ether sent': 257.307,
    'total ether received': 182.2115661,
    'total ether balance': -75.09543387
}

# Convert the input data into a DataFrame (to match model's expected input format)
input_df = pd.DataFrame([input_data])

# Ensure the input data has the same features as the training data
model_features = ['Avg min between sent tnx', 'Avg min between received tnx', 'Time Diff between first and last (Mins)', 
                  'Sent tnx', 'Received Tnx', 'Unique Received From Addresses', 'Unique Sent To Addresses', 
                  'total transactions (including tnx to create contract', 'total Ether sent', 
                  'total ether received', 'total ether balance']  # Replace with the actual feature list from training
input_df = input_df[model_features]

# Make a prediction
prediction = model.predict(input_df)

# Output the prediction (0 or 1 based on FLAG value)
print(f"Predicted FLAG: {prediction[0]}")

# Let's assume you have a larger dataset for visualizations
# If not, you can use the same features to create visualizations, and they will apply to your predicted FLAG.
# Load your dataset again to generate meaningful visualizations

df = pd.read_csv('dataset.csv')  # Load the dataset
df.columns = df.columns.str.strip()  # Clean up column names

# 1. Distribution of Predicted FLAG (0 vs 1)
plt.figure(figsize=(8, 6))
sns.countplot(x=prediction)
plt.title("Distribution of Predicted FLAG")
plt.show()

# 2. Confusion Matrix
y_true = df['FLAG']  # Actual FLAG values
y_pred = model.predict(df[model_features])  # Predicted FLAG values

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 3. Feature Importance
importances = model.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [model_features[i] for i in indices])
plt.title("Feature Importance from Gradient Boosting")
plt.xlabel("Importance")
plt.show()

# 4. Pairplot of Features with Predicted FLAG
df['Predicted_FLAG'] = y_pred
sns.pairplot(df[['Avg min between sent tnx', 'Sent tnx', 'total Ether sent', 'total ether balance', 'Predicted_FLAG']], hue='Predicted_FLAG')
plt.suptitle("Pairplot of Features with Predicted FLAG", y=1.02)
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[model_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.show()

# 6. ROC Curve
fpr, tpr, _ = roc_curve(y_true, model.predict_proba(df[model_features])[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 7. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, model.predict_proba(df[model_features])[:,1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 9. Sent Transactions vs Received Transactions (Colored by Prediction)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sent tnx', y='Received Tnx', hue='Predicted_FLAG', data=df, palette="Set2")
plt.title("Sent Transactions vs Received Transactions (Colored by Predicted FLAG)")
plt.show()

# 10. Boxplot of Time Diff between First and Last Transactions by FLAG
plt.figure(figsize=(8, 6))
sns.boxplot(x='Predicted_FLAG', y='Time Diff between first and last (Mins)', data=df)
plt.title("Boxplot of Time Diff between First and Last Transactions by Predicted FLAG")
plt.show()
