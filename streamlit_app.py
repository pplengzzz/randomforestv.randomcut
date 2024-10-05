import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ด้วย RandomForest', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการค่าระดับน้ำและการพยากรณ์ด้วย RandomForest")

# ฟังก์ชันสำหรับการรวมข้อมูลจากหลายไฟล์
def combine_data(file_1, file_2):
    combined_data = pd.concat([file_1, file_2], axis=0)
    combined_data = combined_data.sort_index()
    return combined_data

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(full_data, train_data):
    filled_data = full_data.copy()
    # คำนวณฟีเจอร์ใหม่ (หากจำเป็น)
    filled_data['week'] = filled_data.index.to_period("W")
    filled_data['month'] = filled_data.index.to_period("M")
    
    missing_weeks = filled_data[filled_data['wl_up'].isna()]['week'].unique()
    for week in missing_weeks:
        week_data = filled_data[filled_data['week'] == week]
        missing_idx = week_data[week_data['wl_up'].isna()].index
        
        # ผสมข้อมูลจากไฟล์ที่ใช้เทรน
        week_train_data = pd.concat([train_data, week_data]).dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])
        
        # กรณีที่มีข้อมูลในสัปดาห์ที่เป็นค่าเต็มเพียงพอ
        if len(week_train_data) > 1:
            X_train = week_train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            y_train = week_train_data['wl_up']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        else:
            # กรณีข้อมูลสัปดาห์นั้นไม่พอ ให้ใช้ข้อมูลจากสัปดาห์ก่อนหน้า
            prev_week = pd.Period(week, freq='W') - 1
            prev_week_data = filled_data[filled_data['week'] == prev_week.strftime('%Y-%m-%d')]
            prev_train_data = pd.concat([train_data, prev_week_data]).dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])
            if len(prev_train_data) > 1:
                X_train = prev_train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                y_train = prev_train_data['wl_up']
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
            else:
                # ถ้าข้อมูลในสัปดาห์ก่อนหน้าก็ไม่พอ ให้ใช้ข้อมูลในเดือนเดียวกัน
                current_month = filled_data.loc[week_data.index[0], 'month']
                month_data = filled_data[(filled_data['month'] == current_month) & (filled_data['week'] != week)]
                month_train_data = pd.concat([train_data, month_data]).dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])
                if len(month_train_data) > 1:
                    X_train = month_train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                    y_train = month_train_data['wl_up']
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                else:
                    continue  # ถ้าหาข้อมูลสำหรับเทรนไม่ได้ ให้ข้ามการเติมค่าไป

        # เติมค่าที่หายไปในสัปดาห์นั้น
        X_missing = week_data.loc[missing_idx, ['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
        X_missing_clean = X_missing.dropna()
        if not X_missing_clean.empty:
            filled_values = model.predict(X_missing_clean)
            filled_data.loc[X_missing_clean.index, 'wl_up'] = filled_values
    return filled_data

# อัปโหลดไฟล์ CSV สำหรับสถานีที่ต้องการเติมข้อมูล
uploaded_file_target = st.file_uploader("เลือกไฟล์ CSV สถานีเป้าหมายที่ต้องการเติมข้อมูล", type="csv")

if uploaded_file_target is not None:
    # อ่านไฟล์สถานีที่ต้องการเติมข้อมูล
    data_target = pd.read_csv(uploaded_file_target)
    data_target['datetime'] = pd.to_datetime(data_target['datetime'])
    data_target.set_index('datetime', inplace=True)
    data_target['hour'] = data_target.index.hour
    data_target['day_of_week'] = data_target.index.dayofweek
    data_target['minute'] = data_target.index.minute
    data_target['lag_1'] = data_target['wl_up'].shift(1)
    data_target['lag_2'] = data_target['wl_up'].shift(2)

    # อัปโหลดไฟล์สำหรับสถานีข้างบนและข้างล่าง
    uploaded_file_1 = st.file_uploader("เลือกไฟล์ CSV สถานีข้างบน (ถ้ามี)", type="csv")
    uploaded_file_2 = st.file_uploader("เลือกไฟล์ CSV สถานีข้างล่าง (ถ้ามี)", type="csv")

    data_train = pd.DataFrame()

    if uploaded_file_1 is not None:
        data_1 = pd.read_csv(uploaded_file_1)
        data_1['datetime'] = pd.to_datetime(data_1['datetime'])
        data_1.set_index('datetime', inplace=True)
        data_1['hour'] = data_1.index.hour
        data_1['day_of_week'] = data_1.index.dayofweek
        data_1['minute'] = data_1.index.minute
        data_1['lag_1'] = data_1['wl_up'].shift(1)
        data_1['lag_2'] = data_1['wl_up'].shift(2)
        data_train = combine_data(data_train, data_1)

    if uploaded_file_2 is not None:
        data_2 = pd.read_csv(uploaded_file_2)
        data_2['datetime'] = pd.to_datetime(data_2['datetime'])
        data_2.set_index('datetime', inplace=True)
        data_2['hour'] = data_2.index.hour
        data_2['day_of_week'] = data_2.index.dayofweek
        data_2['minute'] = data_2.index.minute
        data_2['lag_1'] = data_2['wl_up'].shift(1)
        data_2['lag_2'] = data_2['wl_up'].shift(2)
        data_train = combine_data(data_train, data_2)

    # ตรวจสอบว่ามีข้อมูลสำหรับการเทรนหรือไม่
    if not data_train.empty:
        # เติมค่าที่ขาดหายไป
        filled_data = fill_missing_values(data_target, data_train)

        # แสดงผลลัพธ์เป็นตาราง
        st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
        st.write(filled_data[['wl_up']])
    else:
        st.warning("กรุณาอัปโหลดไฟล์สถานีข้างบนหรือสถานีข้างล่างอย่างน้อยหนึ่งไฟล์เพื่อใช้ในการเทรน")














