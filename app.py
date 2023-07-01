import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Membaca dataset
df = pd.read_csv('framingham.csv')

# Menghapus baris dengan nilai yang hilang
df = df.dropna()

# Memisahkan fitur dan target
features = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
target = 'TenYearCHD'
X = df[features]
y = df[target]

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat objek Gaussian Naive Bayes
gnb = GaussianNB()

# Melatih model dengan data latih
gnb.fit(X_train, y_train)

# Memprediksi nilai target untuk data uji
y_pred = gnb.predict(X_test)

# Menghitung nilai Akurasi, Presisi, Recall, dan F-measure
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f_measure = f1_score(y_test, y_pred)

# Membuat tampilan menggunakan Streamlit
st.title('Hasil Evaluasi Model')
st.write('---')
st.write(f'Akurasi   : {accuracy:.2f}')
st.write(f'Presisi   : {precision:.2f}')
st.write(f'Recall    : {recall:.2f}')
st.write(f'F-measure : {f_measure:.2f}')

# Menampilkan penjelasan hasil pembahasan
st.write('**Penjelasan Hasil Pembahasan:**')
st.write('- Akurasi menggambarkan seberapa baik model memprediksi dengan benar kasus positif dan negatif secara keseluruhan.')
st.write('- Presisi menggambarkan seberapa baik model memprediksi dengan benar kasus positif dari total kasus yang diprediksi positif.')
st.write('- Recall menggambarkan seberapa baik model dapat menemukan kembali (mendeteksi) kasus positif dari total kasus yang sebenarnya positif.')
st.write('- F-measure menggabungkan presisi dan recall dalam satu metrik, menggambarkan keseimbangan antara keduanya.')
st.write('---')


# Visualisasi histogram untuk kolom 'age'
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='TenYearCHD', kde=True)
plt.title('Distribusi Umur dengan CHD')
plt.xlabel('Umur')
plt.ylabel('Frekuensi')
st.pyplot()

# Menambahkan penjelasan plot histogram
st.write('**Plot Histogram: Distribusi Umur dengan CHD**')
st.write('Plot histogram di atas menunjukkan distribusi umur pasien dalam dataset, dengan pemisahan warna berdasarkan keberadaan risiko penyakit kardiovaskular (CHD - TenYearCHD). Histogram memberikan gambaran visual tentang sebaran umur pasien, dengan kurva kernel density estimation (KDE) yang menunjukkan perkiraan kepadatan distribusi.')
st.write('---')

# Menghitung jumlah prediksi benar dan salah
true_positive = sum((y_pred == 1) & (y_test == 1))
true_negative = sum((y_pred == 0) & (y_test == 0))
false_positive = sum((y_pred == 1) & (y_test == 0))
false_negative = sum((y_pred == 0) & (y_test == 1))

# Menghitung akurasi dan presisi dalam bentuk persentase
accuracy_percent = accuracy * 100
precision_percent = precision * 100

# Menampilkan akurasi dan presisi keberhasilan dalam persentase
st.write('**Keberhasilan Model:**')
st.write(f'Akurasi   : {accuracy_percent:.2f}%')
st.write(f'Presisi   : {precision_percent:.2f}%')
st.write('---')



