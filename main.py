# ============================================================
# PREDIKSI KESEJAHTERAAN MASYARAKAT
# Menggunakan Random Forest + SMOTE (versi otomatis)
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1️⃣ Membaca dataset
# ------------------------------------------------------------
print("🔹 Membaca dataset...")
df = pd.read_excel("Data Sosial Kec. Cilawu.xlsx")  # pastikan nama file sesuai

print("Jumlah data:", len(df))
print("Kolom:", list(df.columns))
print(df.head())

# ------------------------------------------------------------
# 2️⃣ Membuat kolom target otomatis (Kesejahteraan)
# ------------------------------------------------------------
bantuan_cols = [
    'Penerima BPNT', 'Penerima BPUM', 'Penerima BST', 'Penerima PKH',
    'Penerima SEMBAKO', 'Penerima Prakerja', 'Penerima KUR',
    'Penerima PKH 2023 (HIMBARA)', 'Penerima SEMBAKO 2023 (HIMBARA)',
    'Keluarga Penerima PKH 2023 (HIMBARA)', 'Keluarga Penerima SEMBAKO (HIMBARA 2023)'
]

# Hitung berapa banyak bantuan "Ya" per individu
df['Jumlah_Bantuan'] = (df[bantuan_cols] == 'Ya').sum(axis=1)

# Tentukan tingkat kesejahteraan berdasarkan jumlah bantuan
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

# ------------------------------------------------------------
# 3️⃣ Siapkan data untuk training
# ------------------------------------------------------------
target_column = "Kesejahteraan"
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode kolom kategori (string) menjadi numerik
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target
y = LabelEncoder().fit_transform(y.astype(str))

# ------------------------------------------------------------
# 4️⃣ Bagi data train-test
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 5️⃣ Atasi ketidakseimbangan dengan SMOTE
# ------------------------------------------------------------
print("\n⚖️  Menyeimbangkan data dengan SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Sebelum SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("Sesudah SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ------------------------------------------------------------
# 6️⃣ Latih model Random Forest
# ------------------------------------------------------------
print("\n🌲 Melatih model Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_res, y_train_res)

# ------------------------------------------------------------
# 7️⃣ Evaluasi hasil
# ------------------------------------------------------------
print("\n📊 Evaluasi model...")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rendah", "Menengah", "Tinggi"]))

# ------------------------------------------------------------
# 8️⃣ Visualisasi Confusion Matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Rendah", "Menengah", "Tinggi"],
    yticklabels=["Rendah", "Menengah", "Tinggi"]
)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest (dengan SMOTE)")
plt.show()

print("\n✅ Selesai! Model sudah dievaluasi.")
