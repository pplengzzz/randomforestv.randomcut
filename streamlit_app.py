import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ด้วย RandomForest', page_icon=':ocean:')

# ชื่อของแอป
st.title("และการพยากรณ์ด้วย RandomForest")

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลหลังตัดค่า (คงเดิม)
def plot_original_data(data, original_nan_indexes=None):
    data = data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot

    plt.figure(figsize=(18, 10))
    
    # Plot ค่าจริง (สีน้ำเงิน)
    plt.plot(data.index, data['wl_up'], label='Actual Values', color='blue', alpha=0.6)
    
    # Plot ค่าที่ถูกตัดออก (สีส้ม)
    if original_nan_indexes is not None:
        plt.plot(original_nan_indexes, data.loc[original_nan_indexes, 'wl_up'], 
                 color='orange', label='Missing Values (Cut)', linestyle='', marker='o')

    # ปรับแต่งกราฟ
    plt.title('Water Level Over Time (After Cutting)', fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Water Level (wl_up)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    st.pyplot(plt)

# ฟังก์ชันสำหรับการแสดงกราฟที่ถูกเติมค่าแล้ว (เปลี่ยนเฉพาะส่วนที่เติมค่า)
def plot_filled_data(original_data, filled_data, original_nan_indexes):
    filled_data = filled_data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot

    plt.figure(figsize=(18, 10))
    
    # Plot ค่าจริง (สีน้ำเงิน)
    plt.plot(original_data.index, original_data['wl_up'], label='Actual Values', color='blue', alpha=0.6)
    
    # Plot ค่าที่เติมด้วยโมเดล (สีเขียวและเป็นเส้นธรรมดา)
    if original_nan_indexes is not None:
        plt.plot(filled_data.loc[original_nan_indexes].index, filled_data.loc[original_nan_indexes, 'wl_up'], 
                 label='Filled Values (Model)', color='green', linestyle='-', linewidth=2, alpha=0.9)

    # ปรับแต่งกราฟ
    plt.title('Water Level Over Time (After Filling)', fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Water Level (wl_up)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    st.pyplot(plt)

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(full_data):
    filled_data = full_data.copy()

    # เพิ่มคอลัมน์ 'week' ให้กับ full_data (สร้างจาก datetime index)
    filled_data['week'] = filled_data.index.to_period("W")

    # เติมค่าในแต่ละอาทิตย์ที่มีข้อมูลขาดหาย
    missing_weeks = filled_data[filled_data['wl_up'].isna()]['week'].unique()

    for week in missing_weeks:
        week_data = filled_data[filled_data['week'] == week]
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

# ฟังก์ชันคำนวณความแม่นยำ
def calculate_accuracy(filled_data, original_data, original_nan_indexes):
    actual_values = original_data.loc[original_nan_indexes, 'wl_up']
    predicted_values = filled_data.loc[original_nan_indexes, 'wl_up']
    
    # คำนวณ MAE และ RMSE
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    # โหลดข้อมูลจริง
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # ทำให้ datetime เป็น tz-naive (ไม่มี timezone)
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    
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

    # กรองข้อมูลที่มีค่า wl_up น้อยกว่า 100 ออก
    filtered_data = data[data['wl_up'] >= 100]

    # แสดงตัวอย่างข้อมูลหลังกรอง
    st.subheader('กราฟตัวอย่างข้อมูลหลังจากกรองค่า')
    plot_original_data(filtered_data)

    # ให้ผู้ใช้เลือกช่วงวันที่ที่ต้องการตัดข้อมูล
    st.subheader("เลือกช่วงวันที่ที่ต้องการตัดข้อมูล")
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(filtered_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(filtered_data.index.max()).date())

    # รวมวันและเวลาที่เลือกเข้าด้วยกันเป็นช่วงเวลา
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)  # ให้ครอบคลุมทั้งวันสิ้นสุด

    # ตรวจสอบว่ามีข้อมูลในช่วงวันที่ที่เลือกหรือไม่
    if not filtered_data.index.isin(pd.date_range(start=start_datetime, end=end_datetime)).any():
        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
    else:
        if st.button("ตัดข้อมูล"):
            # ตัดข้อมูลตามช่วงวันที่ที่ผู้ใช้เลือก
            original_data = filtered_data.copy()

            # ตรวจสอบว่ามีข้อมูลในช่วงวันที่หรือไม่
            date_mask = (filtered_data.index >= start_datetime) & (filtered_data.index <= end_datetime)
            if date_mask.any():
                filtered_data.loc[date_mask, 'wl_up'] = np.nan

                # เก็บตำแหน่ง NaN ก่อนเติมค่า
                original_nan_indexes = filtered_data[filtered_data['wl_up'].isna()].index

                # แสดงกราฟข้อมูลที่ถูกตัด (คงเดิม)
                st.subheader('กราฟข้อมูลหลังจากตัดค่าออก')
                plot_original_data(filtered_data, original_nan_indexes=original_nan_indexes)

                # เติมค่าด้วย RandomForest
                filled_data = fill_missing_values(filtered_data)

                # คำนวณความแม่นยำระหว่างค่าจริงที่ถูกตัดออกกับค่าที่โมเดลเติมกลับ
                st.subheader('ผลการคำนวณความแม่นยำ')
                calculate_accuracy(filled_data, original_data, original_nan_indexes)

                # แสดงกราฟข้อมูลที่เติมค่าด้วยโมเดล RandomForest (เปลี่ยนกราฟ)
                st.subheader('กราฟผลลัพธ์การเติมค่า')
                plot_filled_data(original_data, filled_data, original_nan_indexes)

                # แสดงผลลัพธ์การเติมค่าเป็นตาราง
                st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
                st.write(filled_data[['wl_up']])
            else:
                st.error("ไม่พบข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")





