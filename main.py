import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

FILE_NAME = "data Sosial Kec. Cilawu.xlsx"
RANDOM_STATE = 42
K_RANGE_ELBOW = range(1, 11) # Uji K dari 1 hingga 10

print("🔹 Membaca dataset...")
try:
    df = pd.read_excel(FILE_NAME)
except FileNotFoundError:
    print(f"ERROR: File '{FILE_NAME}' tidak ditemukan")
    exit()
    
print("Jumlah data:", len(df))
print("Kolom;", list(df.columns))

# Membuat kolom target otomatis (Kesejahteraan)
bantuan_cols = [
    'Penerima BPNT', 'Penerima BPUM', 'Penerima BST', 'Penerima PKH',
    'Penerima SEMBAKO', 'Penerima Prakerja', 'Penerima KUR',
    'Penerima PKH 2023 (HIMBARA)', 'Penerima SEMBAKO 2023 (HIMBARA)',
    'Keluarga Penerima PKH 2023 (HIMBARA)', 'Keluarga Penerima SEMBAKO (HIMBARA 2023)'
]

# Hitung banyak bantuan "Ya" per individu
existing_bantuan_cols = [col for col in bantuan_cols if col in df.columns]
df['Jumlah_Bantuan'] = (df[existing_bantuan_cols] == 'Ya').sum(axis=1)

# Tingkat kesejahteraan berdasarkan jumlah bantuan
def tentukan_kesejahteraan(x):
    if x >= 5:
        return "Rendah"
    elif x >= 2:
        return "Menengah"
    else:
        return "Tinggi"

df['Kesejahteraan'] = df['Jumlah_Bantuan'].apply(tentukan_kesejahteraan)

print("\n🔸 Distribusi kelas Kesejahteraan:")
print(df['Kesejahteraan'].value_counts())

# Siapkan data untuk training
target_column = "Kesejahteraan"
X = df.drop(columns=[target_column, 'Jumlah_Bantuan'])
y_raw = df[target_column]

object_cols = X.select_dtypes(include=["object"]).columns

print("\n🛠️  Encoding fitur kategori...")
X_encoded = X.copy()
for col in object_cols:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

print("Scaling Fitur Kategori...")
scaler = StandardScaler()
numeric_cols = X_encoded.columns
X_Scaled = X_encoded.copy()
X_Scaled[numeric_cols] = scaler.fit_transform(X_Scaled[numeric_cols])

# Analisis Clustering Menggunakan Metode Elbow (K-Means)
print("Menerapkan Metode Elbow untuk K-Means...")
wss = []
for k in K_RANGE_ELBOW:
    try:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto', max_iter=300)
        kmeans.fit(X_Scaled)
        wss.append(kmeans.inertia_)
    except Exception as e:
        print(f"Error saat menjalankan K-Means dengan k={k}: {e}")
        wss.append(None)

plt.figure(figsize=(8, 5))
plt.plot(K_RANGE_ELBOW, wss, marker='o', linestyle='--', color='red')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster (k)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WSS / Intertia')
plt.xticks(K_RANGE_ELBOW)
plt.grid(True)
plt.show()
print("Visualisasi Elbow Method Selesai. Amati titik 'siku' pada plot.")

# Klasifikasi Kesejahteraan dengan Random Forest (Logika Asli)
print("Bagian Klasifikasi Kesejahteraan (Menggunakan Label Buatan)")

le = LabelEncoder().fit(y_raw.astype(str))
y_encoded = le.transform(y_raw.astype(str))
target_names_map = le.inverse_transform(sorted(np.unique(y_encoded)))

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# Atasi ketidakseimbangan dengan SMOTE
print("\n⚖️  Menyeimbangkan data dengan SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Sebelum SMOTE:", {dict(zip(target_names_map, pd.Series(y_train_res).value_counts().sort_index()))})
print("Sesudah SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# Latih model Random Forest
print("\n🌲 Melatih model Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
model.fit(X_train_res, y_train_res)

# Evaluasi hasil
print("\n📊 Evaluasi model...")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names_map))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names_map,
    yticklabels=target_names_map
)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest (dengan SMOTE)")
plt.show()

print("\n✅ Selesai! Model sudah dievaluasi.")
