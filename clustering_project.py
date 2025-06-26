import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Set style untuk visualisasi yang lebih baik
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("PROYEK CLUSTERING - ANALISIS DATA TRANSAKSI BANK")
print("=" * 60)


# 1. IMPORT LIBRARY yang di butuhkan 

print("\n1. IMPORT LIBRARY berhasil")


# 2. MEMUAT DATASET

print("\n2. MEMUAT DATASET ")


try:
    df = pd.read_csv('bank_transactions_data_edited.csv')
    print("Dataset berhasil dimuat")
    print(f"- Jumlah baris: {df.shape[0]:,}")
    print(f"- Jumlah kolom: {df.shape[1]}")
    print(f"- Ukuran file: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
except FileNotFoundError:
    print("Error: File 'bank_transactions_data_edited.csv' tidak ditemukan")
    exit()
except Exception as e:
    print(f" Error saat memuat dataset: {e}")
    exit()


# 3. EXPLORATORY DATA ANALYSIS (EDA)

print("\n3. EXPLORATORY DATA ANALYSIS (EDA)")


# 3.1 Informasi Dasar Dataset
print("\n3.1 Informasi Dasar Dataset")
print("Informasi dataset:")
print(df.info())

# 3.2 Preview Data
print("\n3.2 Preview Data")
print("5 baris pertama dataset:")
print(df.head())

print("\n5 baris terakhir dataset:")
print(df.tail())

# 3.3 Statistik Deskriptif
print("\n3.3 Statistik Deskriptif")
print("Statistik deskriptif untuk kolom numerik:")
print(df.describe())

# 3.4 Analisis Nilai Hilang dan Duplikat
print("\n3.4 Analisis Nilai Hilang dan Duplikat")


# Mengecek nilai hilang menggunakan isnull().sum()
print("Mengecek nilai hilang menggunakan isnull().sum():")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Kolom': missing_values.index,
    'Jumlah_Missing': missing_values.values,
    'Persentase_Missing': missing_percentage.values
})
missing_info = missing_info[missing_info['Jumlah_Missing'] > 0].sort_values('Jumlah_Missing', ascending=False)

if len(missing_info) > 0:
    print("Kolom dengan nilai hilang:")
    print(missing_info)
else:
    print(" Tidak ada nilai hilang dalam dataset")

# Mengecek duplikat menggunakan duplicated().sum()
print(f"\nMengecek duplikat menggunakan duplicated().sum():")
duplicate_count = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicate_count}")

if duplicate_count > 0:
    print("Dataset memiliki baris duplikat ")
    
    print("baris duplikat:")
    print(df[df.duplicated(keep=False)].head())
else:
    print("Tidak ada baris duplikat dalam dataset")

# 3.5 Analisis Tipe Data
print("\n3.5 Analisis Tipe Data")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Kolom Numerik ({len(numerical_cols)}): {numerical_cols}")
print(f"Kolom Kategorikal ({len(categorical_cols)}): {categorical_cols}")

# 3.6 Analisis Distribusi Kolom Numerik
print("\n3.6 Analisis Distribusi Kolom Numerik")


# Membuat subplot untuk distribusi kolom numerik
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribusi Kolom Numerik', fontsize=16)

for i, col in enumerate(numerical_cols[:6]):  # Ambil 6 kolom pertama
    row = i // 3
    col_idx = i % 3
    axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[row, col_idx].set_title(f'Distribusi {col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Frekuensi')

plt.tight_layout()
plt.show()

# 3.7 Analisis Kolom Kategorikal
print("\n3.7 Analisis Kolom Kategorikal")


for col in categorical_cols:
    print(f"\nDistribusi untuk kolom '{col}':")
    value_counts = df[col].value_counts()
    print(f"Jumlah kategori unik: {len(value_counts)}")
    print("Top 5 kategori:")
    print(value_counts.head())

# Visualisasi distribusi kategorikal
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribusi Kolom Kategorikal', fontsize=16)

for i, col in enumerate(categorical_cols[:4]):  # Ambil 4 kolom pertama
    row = i // 2
    col_idx = i % 2
    value_counts = df[col].value_counts().head(10)  # Top 10
    axes[row, col_idx].bar(range(len(value_counts)), value_counts.values)
    axes[row, col_idx].set_title(f'Top 10 {col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Jumlah')
    axes[row, col_idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 3.8 Analisis Korelasi (untuk kolom numerik)
print("\n3.8 Analisis Korelasi")


if len(numerical_cols) > 1:
    correlation_matrix = df[numerical_cols].corr()
    print(correlation_matrix.round(3))
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Heatmap Korelasi Kolom Numerik')
    plt.tight_layout()
    plt.show()
else:
    print("Tidak cukup kolom numerik untuk analisis korelasi")

# 3.9 Analisis Outlier
print("\n3.9 Analisis Outlier")
print("-" * 30)


for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(df)) * 100
    
    print(f"{col}: {len(outliers)} outlier ({outlier_percentage:.2f}%)")


# 4. PEMBERSIHAN DATASET

print("\n4. PEMBERSIHAN DATASET")
print("-" * 30)

# 4.1 Menghapus baris duplikat
print("\n4.1 Menghapus Baris Duplikat")

if duplicate_count > 0:
    df_cleaned = df.drop_duplicates()
    print(f"{duplicate_count} baris duplikat berhasil dihapus")
    print(f"Dataset setelah penghapusan duplikat: {df_cleaned.shape[0]} baris x {df_cleaned.shape[1]} kolom")
else:
    df_cleaned = df.copy()
    print("Tidak ada baris duplikat yang perlu dihapus")

# 4.2 Menghapus kolom ID yang tidak relevan
print("\n4.2 Menghapus Kolom ID")


# Menghapus kolom yang memiliki keterangan ID seperti TransactionID, AccountID, DeviceID, IPAddress, MerchantID
columns_to_drop = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address',
                   'MerchantID', 'TransactionDate', 'PreviousTransactionDate']

df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_drop if col in df_cleaned.columns])
print(f" Kolom yang dihapus: {list(set(columns_to_drop) & set(df_cleaned.columns))}")
print(f" Dataset setelah pembersihan: {df_cleaned.shape[0]} baris x {df_cleaned.shape[1]} kolom")

# Update daftar kolom setelah pembersihan
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

print(f"Kolom numerik setelah pembersihan: {numerical_cols}")
print(f"Kolom kategorikal setelah pembersihan: {categorical_cols}")


# 5. DATA PREPROCESSING

print("\n5. DATA PREPROCESSING")
print("-" * 30)

# 5.1 Penanganan Nilai Hilang
print("\n5.1 Penanganan Nilai Hilang")



for col in numerical_cols:
    if df_cleaned[col].isnull().any():
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"   {col}: {df_cleaned[col].isnull().sum()} missing values diisi dengan median: {median_val:.2f}")

for col in categorical_cols:
    if df_cleaned[col].isnull().any():
        mode_val = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(mode_val, inplace=True)
        print(f"   {col}: {df_cleaned[col].isnull().sum()} missing values diisi dengan mode: {mode_val}")

if df_cleaned.isnull().sum().sum() == 0:
    print("Tidak ada nilai hilang setelah penanganan")
else:
    print("Masih terdapat nilai hilang setelah penanganan")

# 5.2 Feature Encoding menggunakan LabelEncoder
print("\n5.2 Feature Encoding menggunakan LabelEncoder")


print("Melakukan feature encoding menggunakan LabelEncoder untuk fitur kategorikal...")
df_encoded = df_cleaned.copy()
label_encoders = {}

for col in categorical_cols:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"{col}: {len(le.classes_)} kategori di-encode")

print(f"Dataset setelah LabelEncoder: {df_encoded.shape[0]} baris x {df_encoded.shape[1]} kolom")

# Update daftar kolom setelah encoding
numerical_cols_encoded = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
print(f" Semua kolom sekarang numerik: {numerical_cols_encoded}")

# 5.3 Feature Scaling
print("\n5.3 Feature Scaling")


print("Melakukan feature scaling ")
features_to_scale = df_encoded.columns


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)
df_scaled = pd.DataFrame(df_scaled, columns=features_to_scale)


print("\nVerifikasi hasil scaling:")
print(f"Rata-rata setelah scaling: {df_scaled.mean().mean():.6f}")
print(f"Standar deviasi setelah scaling: {df_scaled.std().mean():.6f}")

X = df_scaled
print(f" Data siap untuk clustering: {X.shape[0]} sampel x {X.shape[1]} fitur")


# 6. MENENTUKAN JUMLAH CLUSTER OPTIMAL (dengan KElbowVisualizer)

print("\n6. MENENTUKAN JUMLAH CLUSTER OPTIMAL (KElbowVisualizer)")


print("Visualisasi Elbow Method menggunakan KElbowVisualizer...")
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10), timings=False)
visualizer.fit(X)
visualizer.show()

optimal_clusters = visualizer.elbow_value_
print(f" Jumlah cluster optimal hasil KElbowVisualizer: {optimal_clusters}")


# 7. MEMBANGUN MODEL CLUSTERING

print("\n7. MEMBANGUN MODEL CLUSTERING")


print(f"Membangun model K-Means dengan {optimal_clusters} cluster...")
best_model = KMeans(n_clusters=optimal_clusters, init='k-means++',
                max_iter=300, n_init=10, random_state=42)
clusters = best_model.fit_predict(X)

df_encoded['Cluster'] = clusters
df_cleaned_with_clusters = df_cleaned.copy()
df_cleaned_with_clusters['Cluster'] = clusters

print(" Model K-Means berhasil dibangun")
print(f" Inertia (WCSS): {best_model.inertia_:.2f}")


joblib.dump(best_model, 'best_model_clustering')
print(" Model clustering disimpan 'best_model_clustering'")


print("\nDistribusi anggota di setiap cluster:")
cluster_distribution = df_encoded['Cluster'].value_counts().sort_index()
for cluster, count in cluster_distribution.items():
    percentage = (count / len(df_encoded)) * 100
    print(f"  Cluster {cluster}: {count} anggota ({percentage:.1f}%)")


plt.figure(figsize=(10, 6))
cluster_distribution.plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Distribusi Anggota di Setiap Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Jumlah Anggota', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)


for i, v in enumerate(cluster_distribution.values):
    plt.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nBeberapa baris pertama dataset dengan label cluster:")
print(df_cleaned_with_clusters[['TransactionAmount', 'CustomerAge', 'AccountBalance', 'Cluster']].head(10))


# 8. EVALUASI MODEL

print("\n8. EVALUASI MODEL")
print("-" * 30)

# 8.1 Silhouette Score
print("\n8.1 Silhouette Score")


if optimal_clusters > 1:
    silhouette_avg = silhouette_score(X, clusters)
    print(f" Silhouette Score: {silhouette_avg:.4f}")
    
    if silhouette_avg > 0.7:
        print("Kualitas clustering: Sangat Baik")
    elif silhouette_avg > 0.5:
        print("Kualitas clustering: Baik")
    elif silhouette_avg > 0.25:
        print("Kualitas clustering: Cukup")
    else:
        print("Kualitas clustering: Buruk")
else:
    print("Tidak dapat menghitung Silhouette Score untuk 1 cluster")

# 8.2 Interpretasi Hasil
print("\n8.2 Interpretasi Hasil")


print("Karakteristik setiap cluster berdasarkan rata-rata fitur:")
cluster_centers_df = df_encoded.groupby('Cluster').mean()
print(cluster_centers_df.round(3))


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Karakteristik Cluster Berdasarkan Fitur Utama', fontsize=16)


main_features = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 'TransactionDuration']
if len(main_features) > 4:
    main_features = main_features[:4]

for i, feature in enumerate(main_features):
    if feature in df_cleaned_with_clusters.columns:
        row = i // 2
        col = i % 2
        cluster_means = df_cleaned_with_clusters.groupby('Cluster')[feature].mean()
        
        axes[row, col].bar(cluster_means.index, cluster_means.values, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[row, col].set_title(f'Rata-rata {feature} per Cluster')
        axes[row, col].set_xlabel('Cluster')
        axes[row, col].set_ylabel(f'Rata-rata {feature}')
        axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# RINGKASAN HASIL

print("\n" + "=" * 60)
print("RINGKASAN HASIL CLUSTERING")
print("=" * 60)

print(f" Dataset: {df_cleaned_with_clusters.shape[0]:,} transaksi bank")
print(f" Fitur yang digunakan: {X.shape[1]} fitur")
print(f" Jumlah cluster optimal: {optimal_clusters}")
print(f" Silhouette Score: {silhouette_avg:.4f}" if optimal_clusters > 1 else " Silhouette Score: Tidak dapat dihitung")

print("\nKarakteristik Cluster:")
for cluster in range(optimal_clusters):
    cluster_data = df_cleaned_with_clusters[df_cleaned_with_clusters['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} anggota):")
    print(f"  - Rata-rata jumlah transaksi: ${cluster_data['TransactionAmount'].mean():.2f}")
    print(f"  - Rata-rata umur pelanggan: {cluster_data['CustomerAge'].mean():.1f} tahun")
    print(f"  - Rata-rata saldo akun: ${cluster_data['AccountBalance'].mean():.2f}")

print("\n Analisis clustering selesai!")


# 9. INTERPRETASI HASIL CLUSTERING (INVERSI & AGREGASI)
print("\n9. INTERPRETASI HASIL CLUSTERING (INVERSI & AGREGASI)")



df_inverse = pd.DataFrame(scaler.inverse_transform(df_scaled), columns=df_encoded.columns.drop('Cluster', errors='ignore'))


for col, le in label_encoders.items():
    if col in df_inverse.columns:
        df_inverse[col] = le.inverse_transform(df_inverse[col].astype(int))


if 'Cluster' in df_encoded.columns:
    df_inverse['Cluster'] = df_encoded['Cluster']
else:
    df_inverse['Cluster'] = clusters


agg_dict = {}
for col in df_inverse.columns:
    if col == 'Cluster':
        continue
    if col in numerical_cols:
        agg_dict[col] = ['min', 'max', 'mean']
    elif col in categorical_cols:
        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None

agg_result = df_inverse.groupby('Cluster').agg(agg_dict)


print("\nHasil agregasi fitur per cluster:")
print(agg_result)


def interpret_cluster(cluster_id, row):
    desc = f"Klaster {cluster_id+1} didapatkan karena "
    parts = []
    for col in numerical_cols:
        if col in row:
            minv = row[(col, 'min')]
            maxv = row[(col, 'max')]
            meanv = row[(col, 'mean')]
            parts.append(f"{col} memiliki rata-rata {meanv:.2f} dengan batas minimum {minv:.2f} dan maksimum {maxv:.2f}")
    for col in categorical_cols:
        if (col, '<lambda>') in row:
            modev = row[(col, '<lambda>')]
            parts.append(f"{col} memiliki modus kategori '{modev}'")
    return desc + ", dan ".join(parts) + "."

print("\nInterpretasi tiap cluster:")
for idx, row in agg_result.iterrows():
    print(interpret_cluster(idx, row))


# 10. MEMBANGUN MODEL KLASIFIKASI BERDASARKAN LABEL CLUSTER

print("\n10. MEMBANGUN MODEL KLASIFIKASI BERDASARKAN LABEL CLUSTER")



X_class = df_encoded.drop('Cluster', axis=1)
y_class = df_encoded['Cluster']


X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data uji: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================================
# 11. EKSPOR MODEL DAN DATA UNTUK SUBMISSION BMLP
# ============================================================================
print("\n11. EKSPOR MODEL DAN DATA UNTUK SUBMISSION BMLP")
print("-" * 60)

# Simpan model clustering (KMeans) ke .h5
joblib.dump(best_model, 'model_clustering.h5')
print("✓ Model clustering (KMeans) disimpan ke model_clustering.h5")

# Simpan model klasifikasi Decision Tree ke .h5
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'decision_tree_model.h5')
print("✓ Model klasifikasi Decision Tree disimpan ke decision_tree_model.h5")

# Simpan data hasil clustering (fitur + label cluster)
df_encoded.to_csv('data_clustering.csv', index=False)
print("✓ Data hasil clustering disimpan ke data_clustering.csv")

# Simpan data hasil clustering inverse (opsional)
df_inverse.to_csv('data_clustering_inverse.csv', index=False)
print("✓ Data hasil clustering inverse disimpan ke data_clustering_inverse.csv (opsional)")


