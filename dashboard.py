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

st.set_page_config(page_title="Dashboard Forecasting", layout="wide")

# ====================== #
#      Load Data         #
# ====================== #
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="ISO-8859-1")
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    monthly = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
    monthly.set_index('Order Date', inplace=True)
    monthly = monthly.asfreq('M')
    return monthly

monthly_sales = load_data()

# ====================== #
#     Tampilan Header    #
# ====================== #
st.title("📊 Dashboard Forecasting Penjualan Superstore")
st.caption("Model Perbandingan: SARIMAX vs XGBoost | Dataset: train.csv")

with st.expander("🔍 Lihat Data Penjualan Bulanan"):
    st.dataframe(monthly_sales)

# ====================== #
#     Proses Forecasting #
# ====================== #
with st.spinner("⏳ Sedang memproses model..."):
    train = monthly_sales[:'2016']
    test = monthly_sales['2017':]

    # ---------- SARIMAX ----------
    @st.cache_resource
    def train_sarimax(train_data):
        model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)

    results_sarimax = train_sarimax(train)
    forecast_sarimax = results_sarimax.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    forecast_sarimax = pd.Series(forecast_sarimax, index=test.index)
    rmse_sarimax = np.sqrt(mean_squared_error(test, forecast_sarimax))
    mae_sarimax = mean_absolute_error(test, forecast_sarimax)

    # ---------- XGBoost ----------
    monthly_ml = monthly_sales.copy()
    monthly_ml['month'] = monthly_ml.index.month
    monthly_ml['year'] = monthly_ml.index.year

    X = monthly_ml[['month', 'year']]
    y = monthly_ml['Sales']
    X_train, X_test = X[:'2016'], X['2017':]
    y_train, y_test = y[:'2016'], y['2017':]

    @st.cache_resource
    def train_xgb(X, y):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        return model

    model_xgb = train_xgb(X_train, y_train)
    forecast_xgb = model_xgb.predict(X_test)
    forecast_xgb = pd.Series(forecast_xgb, index=y_test.index)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, forecast_xgb))
    mae_xgb = mean_absolute_error(y_test, forecast_xgb)

# ====================== #
#     Visualisasi        #
# ====================== #
st.subheader("📈 Visualisasi Forecasting")
fig, ax = plt.subplots(figsize=(10, 4))
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

# ====================== #
#     Evaluasi Model     #
# ====================== #
st.subheader("🔢 Metrik Evaluasi Model")
col1, col2 = st.columns(2)
with col1:
    st.metric("SARIMAX RMSE", f"{rmse_sarimax:,.2f}")
    st.metric("SARIMAX MAE", f"{mae_sarimax:,.2f}")
with col2:
    st.metric("XGBoost RMSE", f"{rmse_xgb:,.2f}")
    st.metric("XGBoost MAE", f"{mae_xgb:,.2f}")

# ====================== #
#   Interpretasi Model   #
# ====================== #
st.subheader("🧠 Interpretasi dan Analisis Model")

st.markdown("""
Berdasarkan hasil **forecasting penjualan bulanan**, berikut perbandingan kedua model:

- **SARIMAX**:
    - Lebih cocok untuk data musiman dan tren historis.
    - Performa lebih akurat jika tanpa fitur eksternal.
- **XGBoost**:
    - Cocok jika ingin menambahkan variabel eksternal.
    - Performa cukup baik, tapi lebih rawan overfit bila fitur terbatas.

🔍 **Analisis Visual**:
- SARIMAX (hijau) mengikuti tren dan musiman.
- XGBoost (ungu putus-putus) cenderung responsif terhadap fluktuasi kecil.

---
### 📌 Kesimpulan:
""")

if rmse_sarimax < rmse_xgb and mae_sarimax < mae_xgb:
    st.success("✅ Model Terbaik: SARIMAX")
    st.markdown("""
    Berdasarkan evaluasi metrik RMSE dan MAE, model **SARIMAX** menunjukkan performa yang paling akurat dalam memprediksi penjualan bulanan pada dataset ini.

    SARIMAX sangat cocok digunakan ketika:
    - Fokus hanya pada data historis tanpa variabel eksternal.
    - Data memiliki pola musiman dan tren jangka panjang yang konsisten.

    Oleh karena itu, **SARIMAX direkomendasikan** sebagai model utama untuk forecasting pada kasus ini.
    """)
elif rmse_xgb < rmse_sarimax and mae_xgb < mae_sarimax:
    st.success("✅ Model Terbaik: XGBoost")
    st.markdown("""
    Model **XGBoost** memberikan hasil evaluasi yang lebih baik dibandingkan SARIMAX, baik dari segi RMSE maupun MAE.

    Keunggulan XGBoost:
    - Lebih fleksibel dalam menangkap pola non-linier.
    - Sangat ideal jika ingin menggabungkan fitur-fitur tambahan (seperti diskon, hari libur, atau kategori produk) ke dalam prediksi.

    Dalam konteks ini, meskipun hanya menggunakan fitur waktu (`bulan` dan `tahun`), XGBoost tetap menunjukkan performa unggul.
    """)
else:
    st.info("⚖️ Keduanya Memiliki Performa yang Seimbang")
    st.markdown("""
    Evaluasi menunjukkan bahwa **SARIMAX dan XGBoost memberikan hasil yang hampir serupa** dalam memprediksi penjualan.

    Rekomendasi:
    - Gunakan **SARIMAX** jika fokus pada pola historis dan musiman.
    - Gunakan **XGBoost** jika ingin menambahkan variabel eksternal untuk analisis yang lebih kompleks.

    Pemilihan akhir dapat disesuaikan dengan konteks bisnis dan kebutuhan pengembangan model ke depan.
    """)

st.caption("📚 Catatan: RMSE mengukur rata-rata kesalahan kuadrat, sedangkan MAE mengukur rata-rata selisih absolut. Semakin kecil nilainya, semakin baik performa model.")
