# Dokumentasi Proyek Clustering

Dokumentasi ini menjelaskan setiap langkah yang dilakukan dalam proses analisis.

## Daftar Isi
- [1. Import Library](#1-import-library)
- [2. Memuat Dataset](#2-memuat-dataset)
- [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
- [4. Pembersihan Dataset](#4-pembersihan-dataset)
- [5. Data Preprocessing](#5-data-preprocessing)
- [6. Menentukan Jumlah Cluster (KElbowVisualizer)](#6-menentukan-jumlah-cluster-kelbowvisualizer)
- [7. Membangun Model Clustering & Menyimpan Model](#7-membangun-model-clustering--menyimpan-model)
- [8. Evaluasi Model Clustering](#8-evaluasi-model-clustering)
- [9. Interpretasi Hasil Clustering (Inversi & Agregasi)](#9-interpretasi-hasil-clustering-inversi--agregasi)
- [10. Membangun Model Klasifikasi](#10-membangun-model-klasifikasi)

---

## 1. Import Library
Library yang digunakan:
```python
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
```

## 2. Memuat Dataset
Dataset transaksi bank (`bank_transactions_data_edited.csv`) dimuat ke dalam pandas DataFrame.

## 3. Exploratory Data Analysis (EDA)
EDA dilakukan untuk memahami struktur, karakteristik, dan pola awal dari data, termasuk:
- Melihat beberapa baris pertama data (`df.head()`), info kolom (`df.info()`), dan statistik deskriptif (`df.describe()`).
- Mengecek nilai hilang (`df.isnull().sum()`) dan duplikat (`df.duplicated().sum()`).
- Analisis distribusi fitur numerik dan kategorikal, serta korelasi dan outlier.

## 4. Pembersihan Dataset
- Menghapus baris duplikat.
- Menghapus kolom identifikasi dan tanggal yang tidak relevan (`TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`, `TransactionDate`, `PreviousTransactionDate`).

## 5. Data Preprocessing
- **Penanganan nilai hilang:** Median untuk numerik, modus untuk kategorikal.
- **Encoding:** LabelEncoder untuk fitur kategorikal.
- **Scaling:** StandardScaler (atau MinMaxScaler) untuk semua fitur numerik.

## 6. Menentukan Jumlah Cluster (KElbowVisualizer)
- Menentukan jumlah cluster optimal menggunakan KElbowVisualizer dari yellowbrick.
- Visualisasi Elbow Method untuk memilih jumlah cluster terbaik.

```python
from yellowbrick.cluster import KElbowVisualizer
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10), timings=False)
visualizer.fit(X)
visualizer.show()
optimal_clusters = visualizer.elbow_value_
```

## 7. Membangun Model Clustering & Menyimpan Model
- Membangun model KMeans dengan jumlah cluster optimal.
- Menyimpan model hasil clustering dengan `joblib.dump()` agar dapat digunakan kembali dan dinilai otomatis.

```python
from sklearn.cluster import KMeans
import joblib
best_model = KMeans(n_clusters=optimal_clusters, ...)
best_model.fit(X)
joblib.dump(best_model, 'best_model_clustering')
```

## 8. Evaluasi Model Clustering
- Menghitung Silhouette Score untuk menilai kualitas cluster.
- Analisis distribusi anggota di setiap cluster.

## 9. Interpretasi Hasil Clustering (Inversi & Agregasi)
- **Inversi fitur:** Mengembalikan fitur numerik ke skala aslinya (inverse transform scaler) dan fitur kategorikal ke label aslinya (inverse transform LabelEncoder).
- **Agregasi per cluster:** 
  - Fitur numerik: min, max, mean.
  - Fitur kategorikal: modus (mode).
- **Interpretasi:** Menuliskan karakteristik utama tiap cluster, misal:
  ```
  Klaster 1 didapatkan karena TransactionAmount memiliki rata-rata x dengan batas minimum y dan maksimum z, dan CustomerOccupation memiliki modus kategori 'Student'.
  ```

## 10. Membangun Model Klasifikasi
- Menggunakan dataset hasil preprocessing dengan label cluster sebagai target/kelas.
- Melatih model klasifikasi (misal: RandomForestClassifier) untuk memprediksi cluster berdasarkan fitur.
- Evaluasi model klasifikasi dengan akurasi dan classification report.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_class = df_encoded.drop('Cluster', axis=1)
y_class = df_encoded['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, ...)
clf = RandomForestClassifier(...)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

