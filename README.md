## 📦 Superstore Sales Forecasting Dashboard

Aplikasi dashboard interaktif untuk melakukan **forecasting penjualan bulanan** pada dataset Superstore menggunakan dua pendekatan:

* 🔁 **SARIMAX (Seasonal ARIMA with Exogenous Regressors)**
* 🌳 **XGBoost Regressor (Machine Learning)**

Dibuat menggunakan Python dan **Streamlit**, dashboard ini memungkinkan analisis visual, evaluasi performa model, dan interpretasi hasil secara langsung.

---

### 🧾 Fitur Utama

✅ Load data penjualan Superstore dari file CSV
✅ Visualisasi interaktif tren penjualan aktual dan hasil prediksi
✅ Evaluasi performa model menggunakan **RMSE** dan **MAE**
✅ Interpretasi otomatis model terbaik berdasarkan hasil
✅ Cocok untuk **presentasi, edukasi, maupun analisis bisnis**

---

### 📂 Struktur Folder

```
.
├── dashboard.py          # Aplikasi utama Streamlit
├── train.csv             # Dataset penjualan Superstore
├── README.md             # Dokumentasi proyek
```

---

### 🚀 Cara Menjalankan

1. **Pastikan kamu memiliki Python 3.8+ dan pip**
2. **Install semua dependencies**:

```bash
pip install streamlit pandas numpy matplotlib statsmodels xgboost scikit-learn
```

3. **Jalankan aplikasinya**:

```bash
streamlit run dashboard.py
```

4. Buka link yang muncul di browser (biasanya [http://localhost:8501](http://localhost:8501))

---

### 📊 Model yang Digunakan

#### 1. SARIMAX (Statsmodels)

* Model time series klasik untuk data musiman
* Cocok untuk data historis dengan pola tren dan musim
* Parameter: `(1,1,1)x(1,1,1,12)`

#### 2. XGBoost Regressor

* Model machine learning berbasis pohon keputusan
* Gunakan fitur waktu (`bulan`, `tahun`) sebagai input
* Lebih fleksibel untuk generalisasi data non-musiman

---

### 📈 Metrik Evaluasi

Model dievaluasi dengan dua metrik utama:

| Model   | RMSE                 | MAE                  |
| ------- | -------------------- | -------------------- |
| SARIMAX | Lebih rendah         | Lebih rendah         |
| XGBoost | Sedikit lebih tinggi | Sedikit lebih tinggi |

---

### 💡 Interpretasi Otomatis

Dashboard akan:

* Menjelaskan kapan SARIMAX vs XGBoost lebih tepat digunakan
* Memberikan rekomendasi model terbaik berdasarkan hasil
* Menampilkan catatan edukatif seputar metrik dan model

---

### 📌 Contoh Visualisasi

Dashboard akan menampilkan plot seperti berikut:

* Data train (biru)
* Data aktual/test (oranye)
* Prediksi SARIMAX (hijau)
* Prediksi XGBoost (ungu putus-putus)

---

### 🧠 Saran Pengembangan Selanjutnya

* Menambahkan fitur **upload CSV custom**
* Menambahkan variabel eksternal (kategori produk, diskon, dll.)
* Menggunakan model LSTM atau Prophet untuk perbandingan lebih lanjut
