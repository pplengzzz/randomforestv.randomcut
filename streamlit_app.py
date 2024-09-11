import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (RandomForest)', page_icon=':ocean:')

# ชื่อของแอป
st.title("ทดสอบการจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย RandomForest")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

# ฟังก์ชันสำหรับการอ่านข้อมูลและจัดการข้อมูล
def process_data(file_path):
    data = pd.read_csv(file_path)

    # ตรวจสอบให้แน่ใจว่าคอลัมน์ 'datetime' เป็นแบบ datetime และตั้งค่าให้เป็น index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)  # ทำให้เป็น tz-naive (ไม่มี timezone)
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

    # เพิ่มคอลัมน์ 'week' ให้กับ full_data
    data['week'] = data.index.to_period("W").astype(str)

    return data

# ฟังก์ชันสำหรับการตัดข้อมูลตามวันที่และเวลาที่ผู้ใช้เลือก
def manual_cut_data(data, start_datetime, end_datetime):
    # กรองข้อมูลตามช่วงวันที่และเวลา
    data_in_range = data.loc[start_datetime:end_datetime]

    # ตัดข้อมูลในช่วงที่ผู้ใช้เลือก
    data.loc[start_datetime:end_datetime, 'wl_up'] = np.nan

    return data, data_in_range.index

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(data, original_nan_indexes):
    filled_data = data.copy()

    missing_weeks = filled_data[filled_data['wl_up'].isna()]['week'].unique()

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
                idx_to_fill = filled_data.index.intersection(X_missing_clean.index)
                filled_data.loc[idx_to_fill, 'wl_up'] = filled_values[:len(idx_to_fill)]

    # เติมค่าที่ขาดหายด้วย ffill และ bfill หลังจากใช้ RandomForestRegressor
    filled_data['wl_up'].ffill(inplace=True)
    filled_data['wl_up'].bfill(inplace=True)

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

# การประมวลผลหลังจากอัปโหลดไฟล์
if uploaded_file is not None:
    # อ่านและประมวลผลข้อมูลจากไฟล์
    full_data = process_data(uploaded_file)

    # ให้ผู้ใช้เลือกช่วงวันที่ที่สนใจ
    st.subheader("เลือกช่วงเวลา")
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(full_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(full_data.index.max()).date())

    # ปุ่มยืนยันการเลือกช่วงวันที่
    if st.button("ยืนยันการเลือกช่วงเวลา"):
        # กรองข้อมูลตามช่วงวันที่ที่เลือก
        if start_date <= end_date:
            data_selected = full_data.loc[start_date:end_date]

            # แสดงกราฟข้อมูลที่เลือก
            st.subheader("กราฟข้อมูลในช่วงที่เลือก")
            data_selected = data_selected.sort_index()  # เรียงลำดับก่อนแสดง
            plt.figure(figsize=(18, 10))
            plt.plot(data_selected.index, data_selected['wl_up'], label='Water Level', color='blue', alpha=0.6)
            plt.title('Water Level in Selected Date Range', fontsize=18)
            plt.xlabel('Date', fontsize=16)
            plt.ylabel('Water Level (wl_up)', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, fontsize=14)
            plt.yticks(fontsize=14)
            st.pyplot(plt)

            # ให้ผู้ใช้เลือกวันและเวลาที่ต้องการตัดข้อมูล
            st.subheader("เลือกวันและเวลาที่ต้องการตัดข้อมูล")

            # รวมการเลือกวันและเวลาให้เป็นบล็อคเดียว
            start_date = st.date_input("เลือกวันที่เริ่มต้น", pd.to_datetime(data_selected.index.min()).date())
            start_time = st.time_input("เลือกเวลาเริ่มต้น", pd.to_datetime(data_selected.index.min()).time())
            end_date = st.date_input("เลือกวันที่สิ้นสุด", pd.to_datetime(data_selected.index.max()).date())
            end_time = st.time_input("เลือกเวลาสิ้นสุด", pd.to_datetime(data_selected.index.max()).time())

            # รวมวันและเวลาที่เลือกเข้าด้วยกันเป็น datetime
            start_datetime = pd.to_datetime(f"{start_date} {start_time}")
            end_datetime = pd.to_datetime(f"{end_date} {end_time}")

            if st.button("ตัดข้อมูล"):
                # ตัดข้อมูลตามวันที่และเวลาที่ผู้ใช้เลือก
                data_selected, original_nan_indexes = manual_cut_data(data_selected, start_datetime, end_datetime)

                # เติมค่าและเก็บตำแหน่งของ NaN เดิม
                filled_data = fill_missing_values(data_selected, original_nan_indexes)

                # พล๊อตผลลัพธ์การเติมค่า
                st.markdown("---")
                st.write("ทำนายระดับน้ำและเติมค่าในข้อมูลที่ขาดหาย")

                plot_filled_data(filled_data, data_selected, original_nan_indexes)

                # แสดงผลลัพธ์การเติมค่าเป็นตาราง
                st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
                # แสดงเฉพาะข้อมูลในช่วงเวลาที่เลือกในขั้นตอนที่ 1
                selected_range_data = filled_data.loc[start_date:end_date]
                st.write(selected_range_data[['wl_up']])
        else:
            st.error("กรุณาเลือกช่วงวันที่ที่ถูกต้อง (วันเริ่มต้นต้องน้อยกว่าหรือเท่ากับวันสิ้นสุด)")



