import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (RandomForest)', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย RandomForest")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

# ฟังก์ชันสำหรับการอ่านข้อมูลและจัดการข้อมูล
def process_data(file_path):
    data = pd.read_csv(file_path)

    # ตรวจสอบให้แน่ใจว่าคอลัมน์ 'datetime' เป็นแบบ datetime และตั้งค่าให้เป็น index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # สุ่มตัดข้อมูลทั้งวัน
    np.random.seed(42)
    full_data = data.copy()
    days_to_remove = np.random.choice(full_data.index.to_period('D').unique(), size=2, replace=False)  # เลือก 2 วันสุ่ม
    missing_indexes_full_days = full_data[full_data.index.to_period('D').isin(days_to_remove)].index

    # เก็บตำแหน่ง NaN ก่อนเติมค่า
    original_nan_indexes_full_days = missing_indexes_full_days
    full_data.loc[missing_indexes_full_days, 'wl_up'] = np.nan  # ตัดข้อมูลออกทั้งวัน

    # เพิ่มฟีเจอร์ด้านเวลา
    full_data['hour'] = full_data.index.hour
    full_data['day_of_week'] = full_data.index.dayofweek
    full_data['minute'] = full_data.index.minute

    # เพิ่ม Lag Features
    full_data['lag_1'] = full_data['wl_up'].shift(1)
    full_data['lag_2'] = full_data['wl_up'].shift(2)

    # เติมค่าใน lag features
    full_data['lag_1'].ffill(inplace=True)
    full_data['lag_2'].ffill(inplace=True)

    return full_data, original_nan_indexes_full_days

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(data, original_nan_indexes_full_days):
    filled_data_full_days_rf = data.copy()
    filled_data_full_days_rf['week'] = filled_data_full_days_rf.index.to_period("W")
    missing_weeks = filled_data_full_days_rf[filled_data_full_days_rf['wl_up'].isna()]['week'].unique()

    for week in missing_weeks:
        week_data = data[data['week'] == week]
        missing_idx = week_data[week_data['wl_up'].isna()].index
        train_data = week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])

        if len(train_data) > 1:
            X_train = train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            y_train = train_data['wl_up']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            X_missing = week_data.loc[missing_idx, ['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            X_missing_clean = X_missing.dropna()

            if not X_missing_clean.empty:
                filled_values = model.predict(X_missing_clean)
                idx_to_fill = filled_data_full_days_rf.index.intersection(X_missing_clean.index)
                filled_data_full_days_rf.loc[idx_to_fill, 'wl_up'] = filled_values[:len(idx_to_fill)]

    return filled_data_full_days_rf

# ฟังก์ชันสำหรับการ plot ข้อมูล
def plot_filled_data(filled_data, original_data, original_nan_indexes_full_days):
    plt.figure(figsize=(14, 7))

    # สีน้ำเงินสำหรับค่าจริงที่ไม่ได้ถูกตัด
    plt.plot(filled_data.index, original_data['wl_up'], label='Actual Values', color='blue', alpha=0.6)

    # สีฟ้าอ่อนสำหรับค่าที่ถูกตัดออก แสดงเฉพาะช่วงที่ถูกตัด ไม่ให้ต่อเส้น
    cut_once = False
    for i in range(len(original_nan_indexes_full_days) - 1):
        start = original_nan_indexes_full_days[i]
        end = original_nan_indexes_full_days[i + 1]
        if (end - start).total_seconds() / 60 <= 15:  # ตรวจสอบว่าช่วงเวลาติดกัน (ทุก 15 นาที)
            if not cut_once:
                plt.plot(original_data.loc[start:end].index,
                         original_data.loc[start:end, 'wl_up'], color='lightblue', alpha=0.6, label='Missing Values (Cut)')
                cut_once = True
            else:
                plt.plot(original_data.loc[start:end].index,
                         original_data.loc[start:end, 'wl_up'], color='lightblue', alpha=0.6)

    # สีแดงสำหรับค่าที่ถูกเติมหลังจากถูกตัด แสดงเฉพาะช่วงที่ถูกเติม ไม่ต่อเส้นเมื่อห่างกัน
    filled_once = False
    for i in range(len(original_nan_indexes_full_days) - 1):
        start = original_nan_indexes_full_days[i]
        end = original_nan_indexes_full_days[i + 1]
        if (end - start).total_seconds() / 60 <= 15:  # ตรวจสอบว่าช่วงเวลาติดกัน (ทุก 15 นาที)
            if not filled_once:
                plt.plot(filled_data.loc[start:end].index,
                         filled_data.loc[start:end, 'wl_up'], color='red', alpha=0.6, label='Filled Values')
                filled_once = True
            else:
                plt.plot(filled_data.loc[start:end].index,
                         filled_data.loc[start:end, 'wl_up'], color='red', alpha=0.6)

    # ปรับแต่งสไตล์กราฟ
    plt.title('Water Level Over Time (Actual, Cut, and Filled Data)')
    plt.xlabel('Date')
    plt.ylabel('Water Level (wl_up)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)

# การประมวลผลหลังจากอัปโหลดไฟล์
if uploaded_file is not None:
    # อ่านและประมวลผลข้อมูลจากไฟล์
    full_data, original_nan_indexes_full_days = process_data(uploaded_file)

    # เติมค่าและเก็บตำแหน่งของ NaN เดิม
    filled_data = fill_missing_values(full_data, original_nan_indexes_full_days)

    # พล๊อตผลลัพธ์การเติมค่า
    st.markdown("---")
    st.write("ทำนายระดับน้ำและเติมค่าในข้อมูลที่ขาดหาย")

    plot_filled_data(filled_data, full_data, original_nan_indexes_full_days)

    # แสดงผลลัพธ์การเติมค่าเป็นตาราง
    st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
    st.write(filled_data[['wl_up']])
