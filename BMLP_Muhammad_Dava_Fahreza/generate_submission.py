import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Starting submission file generation...")

# Load data
print("Loading dataset...")
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?gid=2020477971&single=true&output=csv'
df = pd.read_csv(url)

print(f"Dataset loaded with shape: {df.shape}")

# Data preprocessing for clustering
print("Preprocessing data for clustering...")

# Handle missing values
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop ID columns
columns_to_drop = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID']
df_clean = df.drop(columns=columns_to_drop, errors='ignore')

# Scale numeric features
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df_clean[numeric_columns] = scaler.fit_transform(df_clean[numeric_columns])

# Encode categorical features
categorical_columns = df_clean.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

print("Data preprocessing completed")

# Clustering
print("Performing clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_clean)

# Add cluster labels
df_clean['Cluster'] = cluster_labels

# Save clustering results
df_clean.to_csv('data_clustering.csv', index=False)
print("Saved data_clustering.csv")

# Save clustering model
joblib.dump(kmeans, 'model_clustering.h5')
print("Saved model_clustering.h5")

# PCA for visualization (optional)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_clean.drop('Cluster', axis=1))
joblib.dump(pca, 'PCA_model_clustering.h5')
print("Saved PCA_model_clustering.h5")

# Add meaningful cluster labels
cluster_labels_mapping = {
    0: 'Low_Risk_Regular',
    1: 'High_Value_Premium', 
    2: 'Medium_Risk_Standard'
}
df_clean['Cluster_Label'] = df_clean['Cluster'].map(cluster_labels_mapping)
df_clean.to_csv('data_clustering_inverse.csv', index=False)
print("Saved data_clustering_inverse.csv")

# Classification
print("Building classification models...")

# Prepare data for classification
X = df_clean.drop(['Cluster', 'Cluster_Label'], axis=1)
y = df_clean['Cluster']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree (mandatory)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Save Decision Tree model
joblib.dump(dt_model, 'decision_tree_model.h5')
print("Saved decision_tree_model.h5")

# Additional models (optional)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'explore_RandomForest_classification.h5')
print("Saved explore_RandomForest_classification.h5")

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, 'explore_GradientBoosting_classification.h5')
print("Saved explore_GradientBoosting_classification.h5")

# Hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
joblib.dump(best_rf_model, 'tuning_classification.h5')
print("Saved tuning_classification.h5")

# Save best model
joblib.dump(best_rf_model, 'best_model_classification.h5')
print("Saved best_model_classification.h5")

# Print results
print("\n" + "="*50)
print("SUBMISSION FILES GENERATED SUCCESSFULLY")
print("="*50)

# Clustering results
silhouette_avg = silhouette_score(X, df_clean['Cluster'])
print(f"Clustering Silhouette Score: {silhouette_avg:.4f}")

# Classification results
y_pred_best = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best, average='weighted')

print(f"Best Model Accuracy: {accuracy:.4f}")
print(f"Best Model F1-Score: {f1:.4f}")

print("\nRequired files generated:")
print("✓ [Clustering] Submission Akhir BMLP_Your_Name.ipynb")
print("✓ [Klasifikasi] Submission Akhir BMLP_Your_Name.ipynb")
print("✓ model_clustering.h5")
print("✓ decision_tree_model.h5")
print("✓ data_clustering.csv")
print("✓ PCA_model_clustering.h5 (optional)")
print("✓ explore_RandomForest_classification.h5 (optional)")
print("✓ explore_GradientBoosting_classification.h5 (optional)")
print("✓ tuning_classification.h5 (optional)")
print("✓ best_model_classification.h5 (optional)")
print("✓ data_clustering_inverse.csv (optional)")

print("\nNext steps:")
print("1. Complete the notebook templates with the code provided")
print("2. Run all cells to ensure no errors")
print("3. Zip all files into BMLP_Nama-siswa.zip")
print("4. Submit the zip file") 