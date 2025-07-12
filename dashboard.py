# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Title
st.title("📊 Dashboard Forecasting Penjualan Superstore")
st.caption("Model Perbandingan: SARIMAX vs XGBoost | Dataset: train.csv")

# Load Data
df = pd.read_csv("train.csv", encoding="ISO-8859-1")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
monthly_sales.set_index('Order Date', inplace=True)
monthly_sales = monthly_sales.asfreq('M')

# Split Data
train = monthly_sales[:'2016']
test = monthly_sales['2017':]

# SARIMAX
model_sarimax = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                         enforce_stationarity=False, enforce_invertibility=False)
results_sarimax = model_sarimax.fit(disp=False)
forecast_sarimax = results_sarimax.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
forecast_sarimax = pd.Series(forecast_sarimax, index=test.index)

rmse_sarimax = np.sqrt(mean_squared_error(test, forecast_sarimax))
mae_sarimax = mean_absolute_error(test, forecast_sarimax)

# XGBoost
monthly_sales_ml = monthly_sales.copy()
monthly_sales_ml['month'] = monthly_sales_ml.index.month
monthly_sales_ml['year'] = monthly_sales_ml.index.year

X = monthly_sales_ml[['month', 'year']]
y = monthly_sales_ml['Sales']

X_train, X_test = X[:'2016'], X['2017':]
y_train, y_test = y[:'2016'], y['2017':]

model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)
forecast_xgb = model_xgb.predict(X_test)
forecast_xgb = pd.Series(forecast_xgb, index=y_test.index)

rmse_xgb = np.sqrt(mean_squared_error(y_test, forecast_xgb))
mae_xgb = mean_absolute_error(y_test, forecast_xgb)

# Plot Forecasting
st.subheader("📈 Visualisasi Forecasting")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train['Sales'], label='Train', color='blue')
ax.plot(test.index, test['Sales'], label='Actual (Test)', color='orange')
ax.plot(test.index, forecast_sarimax, label='Forecast SARIMAX', color='green')
ax.plot(test.index, forecast_xgb, label='Forecast XGBoost', color='purple', linestyle='--')
ax.set_title("Forecasting Penjualan Superstore: SARIMAX vs XGBoost")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Penjualan")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Metrics
st.subheader("🔢 Metrik Evaluasi Model")

col1, col2 = st.columns(2)
with col1:
    st.metric("SARIMAX RMSE", f"{rmse_sarimax:,.2f}")
    st.metric("SARIMAX MAE", f"{mae_sarimax:,.2f}")
with col2:
    st.metric("XGBoost RMSE", f"{rmse_xgb:,.2f}")
    st.metric("XGBoost MAE", f"{mae_xgb:,.2f}")

# Interpretasi & Kesimpulan
st.subheader("🧠 Interpretasi dan Analisis Model")

st.markdown("""
Berdasarkan hasil **forecasting penjualan bulanan** dari dataset Superstore, kita membandingkan dua model: **SARIMAX** dan **XGBoost**.

Berikut adalah hasil evaluasinya:

- **SARIMAX**:
  - Lebih cocok untuk data deret waktu dengan pola musiman dan tren yang jelas.
  - Menghasilkan **RMSE lebih rendah** dan **MAE lebih rendah**, yang menunjukkan prediksi lebih dekat ke data aktual secara konsisten.

- **XGBoost**:
  - Model machine learning berbasis pohon yang fleksibel dan mampu menangkap pola non-linier.
  - Cocok digunakan jika terdapat **variabel eksternal lain** seperti promosi, cuaca, atau kategori produk yang relevan.
  - Dalam kasus ini, hanya menggunakan fitur `bulan` dan `tahun`, sehingga performanya masih di bawah SARIMAX.

🔍 **Analisis Visual**:
- Garis prediksi SARIMAX (hijau) cukup stabil dan mengikuti tren musiman data aktual.
- Garis prediksi XGBoost (ungu putus-putus) cenderung lebih fleksibel, tetapi terkadang terlalu merespons fluktuasi kecil, menyebabkan over/underestimate.

---

### 📌 Kesimpulan:

""")

if rmse_sarimax < rmse_xgb and mae_sarimax < mae_xgb:
    st.success("✅ Model terbaik: **SARIMAX**")
    st.markdown("""
    SARIMAX memberikan hasil prediksi yang lebih akurat untuk dataset ini. Model ini disarankan apabila fokus hanya pada data historis penjualan tanpa variabel eksternal tambahan.
    """)
elif rmse_xgb < rmse_sarimax and mae_xgb < mae_sarimax:
    st.success("✅ Model terbaik: **XGBoost**")
    st.markdown("""
    XGBoost menunjukkan hasil lebih baik. Model ini cocok untuk memperluas prediksi di masa depan terutama jika ingin menggabungkan faktor eksternal (misalnya: diskon, kategori produk, hari libur).
    """)
else:
    st.info("⚖️ Keduanya memiliki performa yang mirip.")
    st.markdown("""
    Baik SARIMAX maupun XGBoost memberikan hasil prediksi yang seimbang. Pemilihan dapat didasarkan pada kompleksitas data dan kebutuhan fitur eksternal.
    """)

st.caption("📚 Catatan: RMSE mengukur rata-rata kesalahan kuadrat, sedangkan MAE mengukur rata-rata selisih absolut. Semakin kecil nilainya, semakin baik model.")

