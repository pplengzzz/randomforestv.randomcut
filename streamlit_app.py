import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (RandomForest)', page_icon=':ocean:')

# ชื่อของแอป
st.title("ทดสอบการจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย RandomForest")

# ฟังก์ชันสำหรับการคำนวณความแม่นยำ
def calculate_accuracy(filled_data, original_nan_indexes, original_data):
    actual_values = original_data.loc[original_nan_indexes, 'wl_up']
    predicted_values = filled_data.loc[original_nan_indexes, 'wl_up']
    
    # คำนวณความแม่นยำ (Mean Absolute Error - MAE)
    mae = np.mean(np.abs(actual_values - predicted_values))
    st.write(f"Mean Absolute Error (MAE): {mae}")

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(full_data):
    filled_data = full_data.copy()

    # เติมค่าในแต่ละอาทิตย์ที่มีข้อมูลขาดหาย
    filled_data['week'] = full_data.index.to_period("W")
    missing_weeks = full_data[full_data['wl_up'].isna()]['week'].unique()

    for week in missing_weeks:
        week_data = full_data[full_data['week'] == week]
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
                filled_data.loc[X_missing_clean.index, 'wl_up'] = filled_values

    return filled_data

# ฟังก์ชันสำหรับการ plot ข้อมูล
def plot_filled_data(filled_data, original_data, original_nan_indexes):
    filled_data = filled_data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot
    original_data = original_data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot

    plt.figure(figsize=(18, 10))

    # สีน้ำเงินสำหรับค่าจริงที่ไม่ได้ถูกตัด
    plt.plot(original_data.index, original_data['wl_up'], label='Actual Values', color='blue', alpha=0.6)

    # สีส้มสำหรับค่าที่ถูกตัดออก
    plt.plot(original_nan_indexes, original_data.loc[original_nan_indexes, 'wl_up'], 
             color='orange', alpha=0.6, label='Missing Values (Cut)', linestyle='')

    # สีเขียวสำหรับค่าที่ถูกเติมหลังจากถูกตัด
    plt.plot(filled_data.loc[original_nan_indexes].index, 
             filled_data.loc[original_nan_indexes, 'wl_up'], color='green', alpha=0.6, label='Filled Values', linestyle='')

    # ปรับแต่งสไตล์กราฟ
    plt.title('Water Level Over Time (Actual, Cut, and Filled Data)', fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Water Level (wl_up)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    st.pyplot(plt)

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    # โหลดข้อมูลจริง
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # เพิ่มฟีเจอร์ด้านเวลา
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['minute'] = data.index.minute

    # เพิ่ม Lag Features
    data['lag_1'] = data['wl_up'].shift(1)
    data['lag_2'] = data['wl_up'].shift(2)

    # เติมค่าใน lag features
    data['lag_1'].ffill(inplace=True)
    data['lag_2'].ffill(inplace=True)

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจ
    st.subheader("เลือกช่วงเวลา")
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(data.index.min()).date())
    start_time = st.time_input("เลือกเวลาเริ่มต้น", pd.to_datetime(data.index.min()).time())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(data.index.max()).date())
    end_time = st.time_input("เลือกเวลาสิ้นสุด", pd.to_datetime(data.index.max()).time())

    # รวมวันและเวลาที่เลือกเข้าด้วยกันเป็น datetime
    start_datetime = pd.to_datetime(f"{start_date} {start_time}")
    end_datetime = pd.to_datetime(f"{end_date} {end_time}")

    if st.button("ตัดข้อมูล"):
        # เก็บค่าข้อมูลจริงไว้ก่อนทำการตัด
        original_data = data.copy()

        # ตัดข้อมูลตามวันที่และเวลาที่ผู้ใช้เลือก
        data.loc[start_datetime:end_datetime, 'wl_up'] = np.nan

        # เก็บตำแหน่ง NaN ก่อนเติมค่า
        original_nan_indexes = data[data['wl_up'].isna()].index

        # เติมค่าที่ขาดหายไปด้วย RandomForestRegressor
        filled_data = fill_missing_values(data)

        # เติมค่าว่างที่เหลือด้วย ffill และ bfill
        filled_data['wl_up'].ffill(inplace=True)
        filled_data['wl_up'].bfill(inplace=True)

        # คำนวณความแม่นยำ
        calculate_accuracy(filled_data, original_nan_indexes, original_data)

        # แสดงกราฟข้อมูลที่เติมค่า
        st.subheader('กราฟผลลัพธ์การเติมค่า')
        plot_filled_data(filled_data, original_data, original_nan_indexes)

        # แสดงผลลัพธ์การเติมค่าเป็นตาราง
        st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
        st.write(filled_data[['wl_up']])




