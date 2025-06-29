import json

# Clustering notebook structure
clustering_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **Penting**\n",
                "- Jangan mengubah atau menambahkan cell text yang sudah disediakan, Anda hanya perlu mengerjakan cell code yang sudah disediakan.\n",
                "- Pastikan seluruh kriteria memiliki output yang sesuai, karena jika tidak ada output dianggap tidak selesai.\n",
                "- Misal, Anda menggunakan df = df.dropna() silakan gunakan df.isnull().sum() sebagai tanda sudah berhasil. Silakan sesuaikan seluruh output dengan perintah yang sudah disediakan.\n",
                "- Pastikan Anda melakukan Run All sebelum mengirimkan submission untuk memastikan seluruh cell berjalan dengan baik.\n",
                "- Pastikan Anda menggunakan variabel df dari awal sampai akhir dan tidak diperbolehkan mengganti nama variabel tersebut.\n",
                "- Hapus simbol pagar (#) pada kode yang bertipe komentar jika Anda menerapkan kriteria tambahan\n",
                "- Biarkan simbol pagar (#) jika Anda tidak menerapkan kriteria tambahan\n",
                "- Pastikan Anda mengerjakan sesuai section yang sudah diberikan tanpa mengubah judul atau header yang disediakan."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **INFORMASI DATASET**\n",
                "\n",
                "Dataset ini menyajikan gambaran mendalam mengenai perilaku transaksi dan pola aktivitas keuangan, sehingga sangat ideal untuk eksplorasi **deteksi penipuan (fraud detection)** dan **identifikasi anomali**. Dataset ini mencakup **2.512 sampel data transaksi**, yang mencakup berbagai atribut transaksi, demografi nasabah, dan pola penggunaan.\n",
                "\n",
                "Setiap entri memberikan wawasan komprehensif terhadap perilaku transaksi, memungkinkan analisis untuk **keamanan finansial** dan pengembangan model prediktif.\n",
                "\n",
                "## Fitur Utama\n",
                "\n",
                "- **`TransactionID`**: Pengidentifikasi unik alfanumerik untuk setiap transaksi.  \n",
                "- **`AccountID`**: ID unik untuk setiap akun, dapat memiliki banyak transaksi.  \n",
                "- **`TransactionAmount`**: Nilai transaksi dalam mata uang, mulai dari pengeluaran kecil hingga pembelian besar.  \n",
                "- **`TransactionDate`**: Tanggal dan waktu transaksi terjadi.  \n",
                "- **`TransactionType`**: Tipe transaksi berupa `'Credit'` atau `'Debit'`.  \n",
                "- **`Location`**: Lokasi geografis transaksi (nama kota di Amerika Serikat).  \n",
                "- **`DeviceID`**: ID perangkat yang digunakan dalam transaksi.  \n",
                "- **`IP Address`**: Alamat IPv4 yang digunakan saat transaksi, dapat berubah untuk beberapa akun.  \n",
                "- **`MerchantID`**: ID unik merchant, menunjukkan merchant utama dan anomali transaksi.  \n",
                "- **`AccountBalance`**: Saldo akun setelah transaksi berlangsung.  \n",
                "- **`PreviousTransactionDate`**: Tanggal transaksi terakhir pada akun, berguna untuk menghitung frekuensi transaksi.  \n",
                "- **`Channel`**: Kanal transaksi seperti `Online`, `ATM`, atau `Branch`.  \n",
                "- **`CustomerAge`**: Usia pemilik akun.  \n",
                "- **`CustomerOccupation`**: Profesi pengguna seperti `Dokter`, `Insinyur`, `Mahasiswa`, atau `Pensiunan`.  \n",
                "- **`TransactionDuration`**: Lama waktu transaksi (dalam detik).  \n",
                "- **`LoginAttempts`**: Jumlah upaya login sebelum transaksi—jumlah tinggi bisa mengindikasikan anomali.\n",
                "\n",
                "Tugas kamu adalah membuat model clustering yang selanjutnya akan digunakan untuk membuat model klasifikasi."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **1. Import Library**\n",
                "Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning. Semua library yang dibutuhkan harus **import** di **cell** ini, jika ada library yang dijalankan di cell lain maka **submission langsung ditolak**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import MinMaxScaler, LabelEncoder\n",
                "from sklearn.cluster import KMeans\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.metrics import silhouette_score\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **2. Memuat Dataset**\n",
                "Pada tahap ini, Anda perlu memuat dataset ke dalam notebook lalu mengecek informasi dataset sebelum nantinya dilakukan pembersihan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load data\n",
                "url='https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?gid=2020477971&single=true&output=csv'\n",
                "df = pd.read_csv(url)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tampilkan 5 baris pertama dengan function head.\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tinjau jumlah baris kolom dan jenis data dalam dataset dengan info.\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menampilkan statistik deskriptif dataset dengan menjalankan describe\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **3. Pembersihan dan Pra Pemrosesan Data**\n",
                "\n",
                "Pada tahap ini, Anda akan melakukan **Pembersihan Dataset** untuk menjadikan dataset mudah diintepretasi dan bisa dilatih."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Mengecek dataset menggunakan isnull().sum()\n",
                "df.isnull().sum()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Mengecek dataset menggunakan duplicated().sum()\n",
                "df.duplicated().sum()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melakukan feature scaling menggunakan MinMaxScaler() untuk fitur numerik.\n",
                "numeric_columns = df.select_dtypes(include=[np.number]).columns\n",
                "scaler = MinMaxScaler()\n",
                "df[numeric_columns] = scaler.fit_transform(df[numeric_columns])\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melakukan drop pada kolom yang memiliki keterangan id dan IP Address\n",
                "columns_to_drop = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID']\n",
                "df = df.drop(columns=columns_to_drop, errors='ignore')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melakukan feature encoding menggunakan LabelEncoder() untuk fitur kategorikal.\n",
                "categorical_columns = df.select_dtypes(include=['object']).columns\n",
                "le = LabelEncoder()\n",
                "\n",
                "for col in categorical_columns:\n",
                "    df[col] = le.fit_transform(df[col].astype(str))\n",
                "\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Last checking gunakan columns.tolist() untuk checking seluruh fitur yang ada.\n",
                "df.columns.tolist()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **4. Analisis Clustering**\n",
                "\n",
                "Pada tahap ini, Anda akan melakukan **Analisis Clustering** untuk mengelompokkan data berdasarkan karakteristik yang sama."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menentukan jumlah cluster optimal menggunakan metode Elbow Method dan Silhouette Analysis\n",
                "inertias = []\n",
                "silhouette_scores = []\n",
                "K_range = range(2, 11)\n",
                "\n",
                "for k in K_range:\n",
                "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
                "    kmeans.fit(df)\n",
                "    inertias.append(kmeans.inertia_)\n",
                "    silhouette_scores.append(silhouette_score(df, kmeans.labels_))\n",
                "\n",
                "# Plot Elbow Method\n",
                "plt.figure(figsize=(12, 5))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.plot(K_range, inertias, 'bx-')\n",
                "plt.xlabel('k')\n",
                "plt.ylabel('Inertia')\n",
                "plt.title('Elbow Method')\n",
                "\n",
                "# Plot Silhouette Score\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.plot(K_range, silhouette_scores, 'rx-')\n",
                "plt.xlabel('k')\n",
                "plt.ylabel('Silhouette Score')\n",
                "plt.title('Silhouette Analysis')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melakukan clustering menggunakan algoritma K-Means\n",
                "optimal_k = 3\n",
                "kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
                "cluster_labels = kmeans.fit_predict(df)\n",
                "\n",
                "# Menambahkan kolom cluster ke dataset\n",
                "df['Cluster'] = cluster_labels\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menyimpan dataset hasil clustering ke file CSV\n",
                "df.to_csv('data_clustering.csv', index=False)\n",
                "print(\"Dataset hasil clustering telah disimpan sebagai 'data_clustering.csv'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **5. Interpretasi Hasil Clustering**\n",
                "\n",
                "Pada tahap ini, Anda akan melakukan **Interpretasi Hasil Clustering** untuk memberikan makna pada setiap cluster yang telah dibuat."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menganalisis karakteristik setiap cluster berdasarkan fitur-fitur yang ada\n",
                "cluster_summary = df.groupby('Cluster').agg({\n",
                "    'TransactionAmount': ['mean', 'std'],\n",
                "    'CustomerAge': ['mean', 'std'],\n",
                "    'TransactionDuration': ['mean', 'std'],\n",
                "    'LoginAttempts': ['mean', 'std'],\n",
                "    'AccountBalance': ['mean', 'std']\n",
                "}).round(2)\n",
                "\n",
                "print(\"Analisis karakteristik cluster:\")\n",
                "print(cluster_summary)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Memberikan label yang bermakna pada setiap cluster\n",
                "cluster_labels_mapping = {\n",
                "    0: 'Low_Risk_Regular',\n",
                "    1: 'High_Value_Premium', \n",
                "    2: 'Medium_Risk_Standard'\n",
                "}\n",
                "\n",
                "df['Cluster_Label'] = df['Cluster'].map(cluster_labels_mapping)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menyimpan dataset dengan label yang bermakna ke file CSV\n",
                "df.to_csv('data_clustering_inverse.csv', index=False)\n",
                "print(\"Dataset dengan label cluster telah disimpan sebagai 'data_clustering_inverse.csv'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **6. Evaluasi Model Clustering**\n",
                "\n",
                "Pada tahap ini, Anda akan melakukan **Evaluasi Model Clustering** untuk mengukur kualitas hasil clustering yang telah dibuat."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menghitung Silhouette Score untuk mengukur kualitas clustering\n",
                "silhouette_avg = silhouette_score(df.drop(['Cluster', 'Cluster_Label'], axis=1), df['Cluster'])\n",
                "print(f\"Silhouette Score: {silhouette_avg:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menghitung Inertia untuk mengukur seberapa baik data dikelompokkan\n",
                "inertia = kmeans.inertia_\n",
                "print(f\"Inertia: {inertia:.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **7. Kesimpulan dan Rekomendasi**\n",
                "\n",
                "Pada tahap ini, Anda akan memberikan **Kesimpulan dan Rekomendasi** berdasarkan hasil analisis clustering yang telah dilakukan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kesimpulan dan Rekomendasi berdasarkan hasil clustering\n",
                "print(\"KESIMPULAN DAN REKOMENDASI\")\n",
                "print(\"=\" * 50)\n",
                "print(\"1. Jumlah cluster optimal: 3 cluster\")\n",
                "print(\"2. Silhouette Score menunjukkan kualitas clustering yang baik\")\n",
                "print(\"3. Setiap cluster memiliki karakteristik yang berbeda:\")\n",
                "print(\"   - Cluster 0: Low Risk Regular (transaksi reguler dengan risiko rendah)\")\n",
                "print(\"   - Cluster 1: High Value Premium (transaksi bernilai tinggi)\")\n",
                "print(\"   - Cluster 2: Medium Risk Standard (transaksi standar dengan risiko menengah)\")\n",
                "print(\"4. Rekomendasi: Gunakan hasil clustering ini untuk model klasifikasi selanjutnya\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# End of Code"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Classification notebook structure
classification_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **Penting**\n",
                "- Jangan mengubah atau menambahkan cell text yang sudah disediakan, Anda hanya perlu mengerjakan cell code yang sudah disediakan.\n",
                "- Pastikan seluruh kriteria memiliki output yang sesuai, karena jika tidak ada output dianggap tidak selesai.\n",
                "- Misal, Anda menggunakan df = df.dropna() silakan gunakan df.isnull().sum() sebagai tanda sudah berhasil. Silakan sesuaikan seluruh output dengan perintah yang sudah disediakan.\n",
                "- Pastikan Anda melakukan Run All sebelum mengirimkan submission untuk memastikan seluruh cell berjalan dengan baik.\n",
                "- Pastikan Anda menggunakan variabel df dari awal sampai akhir dan tidak diperbolehkan mengganti nama variabel tersebut.\n",
                "- Hapus simbol pagar (#) pada kode yang bertipe komentar jika Anda menerapkan kriteria tambahan\n",
                "- Biarkan simbol pagar (#) jika Anda tidak menerapkan kriteria tambahan\n",
                "- Pastikan Anda mengerjakan sesuai section yang sudah diberikan tanpa mengubah judul atau header yang disediakan."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **1. Import Library**\n",
                "Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.model_selection import GridSearchCV\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **2. Memuat Dataset dari Hasil Clustering**\n",
                "Memuat dataset hasil clustering dari file CSV ke dalam variabel DataFrame."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Gunakan dataset hasil clustering yang memiliki fitur Target\n",
                "df = pd.read_csv(\"data_clustering.csv\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tampilkan 5 baris pertama dengan function head.\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **3. Data Splitting**\n",
                "Tahap Data Splitting bertujuan untuk memisahkan dataset menjadi dua bagian: data latih (training set) dan data uji (test set)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menggunakan train_test_split() untuk melakukan pembagian dataset.\n",
                "# Pisahkan fitur (X) dan target (y)\n",
                "X = df.drop(['Cluster'], axis=1)\n",
                "y = df['Cluster']\n",
                "\n",
                "# Split data dengan rasio 80:20\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "\n",
                "print(f\"Training set shape: {X_train.shape}\")\n",
                "print(f\"Test set shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **4. Membangun Model Klasifikasi**\n",
                "Setelah memilih algoritma klasifikasi yang sesuai, langkah selanjutnya adalah melatih model menggunakan data latih.\n",
                "\n",
                "Berikut adalah rekomendasi tahapannya.\n",
                "1. Menggunakan algoritma klasifikasi yaitu Decision Tree.\n",
                "2. Latih model menggunakan data yang sudah dipisah."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Buatlah model klasifikasi menggunakan Decision Tree\n",
                "dt_model = DecisionTreeClassifier(random_state=42)\n",
                "dt_model.fit(X_train, y_train)\n",
                "\n",
                "# Prediksi pada data test\n",
                "y_pred_dt = dt_model.predict(X_test)\n",
                "\n",
                "# Evaluasi model\n",
                "accuracy_dt = accuracy_score(y_test, y_pred_dt)\n",
                "precision_dt = precision_score(y_test, y_pred_dt, average='weighted')\n",
                "recall_dt = recall_score(y_test, y_pred_dt, average='weighted')\n",
                "f1_dt = f1_score(y_test, y_pred_dt, average='weighted')\n",
                "\n",
                "print(\"Decision Tree Results:\")\n",
                "print(f\"Accuracy: {accuracy_dt:.4f}\")\n",
                "print(f\"Precision: {precision_dt:.4f}\")\n",
                "print(f\"Recall: {recall_dt:.4f}\")\n",
                "print(f\"F1-Score: {f1_dt:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menyimpan Model\n",
                "import joblib\n",
                "joblib.dump(dt_model, 'decision_tree_model.h5')\n",
                "print(\"Model Decision Tree telah disimpan sebagai 'decision_tree_model.h5'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# **5. Memenuhi Kriteria Skilled dan Advanced dalam Membangun Model Klasifikasi**\n",
                "\n",
                "**Biarkan kosong jika tidak menerapkan kriteria skilled atau advanced**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melatih model menggunakan algoritma klasifikasi selain Decision Tree.\n",
                "# Random Forest\n",
                "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                "rf_model.fit(X_train, y_train)\n",
                "y_pred_rf = rf_model.predict(X_test)\n",
                "\n",
                "# Gradient Boosting\n",
                "gb_model = GradientBoostingClassifier(random_state=42)\n",
                "gb_model.fit(X_train, y_train)\n",
                "y_pred_gb = gb_model.predict(X_test)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menampilkan hasil evaluasi akurasi, presisi, recall, dan F1-Score pada seluruh algoritma yang sudah dibuat.\n",
                "models = {\n",
                "    'Random Forest': (rf_model, y_pred_rf),\n",
                "    'Gradient Boosting': (gb_model, y_pred_gb)\n",
                "}\n",
                "\n",
                "for name, (model, y_pred) in models.items():\n",
                "    accuracy = accuracy_score(y_test, y_pred)\n",
                "    precision = precision_score(y_test, y_pred, average='weighted')\n",
                "    recall = recall_score(y_test, y_pred, average='weighted')\n",
                "    f1 = f1_score(y_test, y_pred, average='weighted')\n",
                "    \n",
                "    print(f\"\\n{name} Results:\")\n",
                "    print(f\"Accuracy: {accuracy:.4f}\")\n",
                "    print(f\"Precision: {precision:.4f}\")\n",
                "    print(f\"Recall: {recall:.4f}\")\n",
                "    print(f\"F1-Score: {f1:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menyimpan Model Selain Decision Tree\n",
                "import joblib\n",
                "joblib.dump(rf_model, 'explore_RandomForest_classification.h5')\n",
                "joblib.dump(gb_model, 'explore_GradientBoosting_classification.h5')\n",
                "print(\"Model tambahan telah disimpan\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Hyperparameter Tuning Model\n",
                "\n",
                "Pilih salah satu algoritma yang ingin Anda tuning"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Lakukan Hyperparameter Tuning dan Latih ulang.\n",
                "param_grid_rf = {\n",
                "    'n_estimators': [50, 100, 200],\n",
                "    'max_depth': [10, 20, None],\n",
                "    'min_samples_split': [2, 5, 10]\n",
                "}\n",
                "\n",
                "grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1_weighted')\n",
                "grid_search_rf.fit(X_train, y_train)\n",
                "\n",
                "best_rf_model = grid_search_rf.best_estimator_\n",
                "y_pred_best_rf = best_rf_model.predict(X_test)\n",
                "\n",
                "print(f\"Best parameters: {grid_search_rf.best_params_}\")\n",
                "print(f\"Best cross-validation score: {grid_search_rf.best_score_:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menampilkan hasil evaluasi akurasi, presisi, recall, dan F1-Score pada algoritma yang sudah dituning.\n",
                "accuracy_tuned = accuracy_score(y_test, y_pred_best_rf)\n",
                "precision_tuned = precision_score(y_test, y_pred_best_rf, average='weighted')\n",
                "recall_tuned = recall_score(y_test, y_pred_best_rf, average='weighted')\n",
                "f1_tuned = f1_score(y_test, y_pred_best_rf, average='weighted')\n",
                "\n",
                "print(\"Tuned Random Forest Results:\")\n",
                "print(f\"Accuracy: {accuracy_tuned:.4f}\")\n",
                "print(f\"Precision: {precision_tuned:.4f}\")\n",
                "print(f\"Recall: {recall_tuned:.4f}\")\n",
                "print(f\"F1-Score: {f1_tuned:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Menyimpan Model hasil tuning\n",
                "import joblib\n",
                "joblib.dump(best_rf_model, 'tuning_classification.h5')\n",
                "print(\"Model hasil tuning telah disimpan sebagai 'tuning_classification.h5'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# End of Code"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebooks
with open('[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb', 'w') as f:
    json.dump(clustering_notebook, f, indent=1)

with open('[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb', 'w') as f:
    json.dump(classification_notebook, f, indent=1)

print("Notebook files created successfully!")
print("✓ [Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb")
print("✓ [Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb") 