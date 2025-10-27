import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


print("🔹 Membaca dataset...")
df = pd.read_excel("Data Sosial Kec. Cilawu.xlsx")

bantuan_cols = [
    'Penerima BPNT', 'Penerima BPUM', 'Penerima BST', 'Penerima PKH', 
    'Penerima SEMBAKO', 'Penerima Prakerja', 'Penerima KUR',
    'Penerima PKH 2023 (HIMBARA)', 'Penerima SEMBAKO 2023 (HIMBARA)',
    'Keluarga Penerima PKH 2023 (HIMBARA)', 'Keluarga Penerima SEMBAKO (HIMBARA 2023)'
]

for col in bantuan_cols:
    if col not in df.columns:
        print(f"⚠️ Kolom {col} tidak ditemukan di dataset!")
    else:
        df[col] = df[col].map({'Ya': 1, 'Tidak': 0})

df['jumlah_bantuan'] = df[bantuan_cols].sum(axis=1)

def kategori(row):
    if row >= 6:
        return "Rendah"
    elif row >= 3:
        return "Menengah"
    else:
        return "Tinggi"

df['Kesejahteraan'] = df['jumlah_bantuan'].apply(kategori)

fitur = ['Jenis Kelamin', 'Status Kawin', 'Pekerjaan', 'Status Pekerjaan', 'Pendidikan']
X = df[fitur]
y = df['Kesejahteraan']

le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col])

print("🔹 Melatih model Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== HASIL EVALUASI MODEL ===")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# 6. VISUALISASI PENTINGNYA FITUR
# ==========================
importances = model.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8,4))
plt.barh(feat_names, importances, color='skyblue')
plt.title("Faktor yang Paling Mempengaruhi Kesejahteraan")
plt.xlabel("Tingkat Kepentingan")
plt.tight_layout()
plt.show()

# ==========================
# 7. SIMPAN HASIL PREEDIKSI
# ==========================
df['Prediksi'] = model.predict(X)
df.to_excel("Hasil_Prediksi_Kesejahteraan.xlsx", index=False)
print("\n✅ Hasil prediksi disimpan di: Hasil_Prediksi_Kesejahteraan.xlsx")
