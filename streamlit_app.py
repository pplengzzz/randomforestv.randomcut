import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การจัดการข้อมูลค่าระดับน้ำและพยากรณ์ด้วย RandomForest', page_icon=':ocean:')

# ชื่อของแอป
st.title("และการพยากรณ์ด้วย RandomForest")

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลทั้งเดือนก่อนการตัดค่า
def plot_full_month_data(data, start_datetime):
    # กำหนดเดือนจากวันที่ที่เลือก
    selected_month = start_datetime.to_period("M")
    month_data = data[data.index.to_period("M") == selected_month]
    
    # แสดงกราฟข้อมูลทั้งเดือน
    fig = px.line(month_data, x=month_data.index, y='wl_up', title=f'Water Level in {selected_month} (Full Month)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    
    # ปรับแต่งกราฟ
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Level (wl_up)",
        title_font=dict(size=18),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x",
        legend=dict(itemsizing='constant', orientation='v', xanchor='center', yanchor='top'),
    )
    st.plotly_chart(fig)

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลหลังตัดค่า (ใช้ plotly)
def plot_original_data(data, original_nan_indexes=None):
    data = data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot

    fig = px.line(data, x=data.index, y='wl_up', title='Water Level Over Time (After Cutting)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    
    # Plot ค่าที่ถูกตัดออก (สีส้ม)
    if original_nan_indexes is not None:
        fig.add_scatter(x=original_nan_indexes, y=data.loc[original_nan_indexes, 'wl_up'], mode='markers', name='Missing Values (Cut)', marker=dict(color='orange'))

    # ปรับแต่งกราฟและซ่อน legend สำหรับกราฟแรก
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Level (wl_up)",
        title_font=dict(size=18),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x",
        legend=dict(itemsizing='constant', title_text=' ', orientation='v', xanchor='center', yanchor='top'),
    )
    st.plotly_chart(fig)

# ฟังก์ชันสำหรับการแสดงกราฟที่ถูกเติมค่าแล้ว (ใช้ plotly)
def plot_filled_data(original_data, filled_data, original_nan_indexes):
    original_data = original_data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot
    filled_data = filled_data.sort_index()  # เรียงลำดับ datetime ก่อนการ plot

    fig = px.line(original_data, x=original_data.index, y='wl_up', title='Water Level Over Time (After Filling)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})

    # Plot ค่าที่ถูกตัดออก (สีส้ม)
    if original_nan_indexes is not None:
        fig.add_scatter(x=original_nan_indexes, y=original_data.loc[original_nan_indexes, 'wl_up'], mode='markers', name='Cut Values', marker=dict(color='orange'))

    # Plot ค่าที่เติมด้วยโมเดล (สีเขียว)
    if original_nan_indexes is not None:
        fig.add_scatter(x=filled_data.loc[original_nan_indexes].index, y=filled_data.loc[original_nan_indexes, 'wl_up'], mode='lines', name='Filled Values (Model)', line=dict(color='green'))

    # ปรับแต่งกราฟและลดความเข้มของเส้นจริง (สีน้ำเงิน) ให้โปร่งใสขึ้น
    fig.update_traces(line=dict(color='blue', width=2, dash='solid'), selector=dict(name='Actual Values'))
    fig.update_traces(opacity=0.6, selector=dict(name='Actual Values'))  # เพิ่มความโปร่งใสให้เส้นจริง
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Level (wl_up)",
        title_font=dict(size=18),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x",
        legend=dict(itemsizing='constant', orientation='v'),
    )
    st.plotly_chart(fig)

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor (ปรับให้ใช้ข้อมูลเดือนหากข้อมูลไม่พอ)
def fill_missing_values(full_data):
    filled_data = full_data.copy()

    # เพิ่มคอลัมน์ 'week' และ 'month' ให้กับ full_data (สร้างจาก datetime index)
    filled_data['week'] = filled_data.index.to_period("W")
    filled_data['month'] = filled_data.index.to_period("M")

    # เติมค่าในแต่ละอาทิตย์ที่มีข้อมูลขาดหาย
    missing_weeks = filled_data[filled_data['wl_up'].isna()]['week'].unique()

    for week in missing_weeks:
        week_data = filled_data[filled_data['week'] == week]
        missing_idx = week_data[week_data['wl_up'].isna()].index
        train_data = week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])

        if len(train_data) > 1:
            # ฝึกโมเดลจากข้อมูลสัปดาห์ปัจจุบัน
            X_train = train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            y_train = train_data['wl_up']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        else:
            # ถ้าไม่มีข้อมูลเพียงพอในสัปดาห์นี้ ให้ลองใช้ข้อมูลสัปดาห์ก่อนหน้า
            prev_week = pd.Period(week, freq='W') - 1
            prev_week_data = filled_data[filled_data['week'] == prev_week.strftime('%Y-%m-%d')]
            
            if len(prev_week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])) > 1:
                X_train = prev_week_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                y_train = prev_week_data['wl_up']

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
            else:
                # ถ้าไม่มีข้อมูลในสัปดาห์ก่อนหน้า ให้ใช้ข้อมูลทั้งเดือน
                current_month = filled_data.loc[week_data.index[0], 'month']
                month_data = filled_data[(filled_data['month'] == current_month) & (filled_data['week'] != week)]
                month_train_data = month_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])

                if len(month_train_data) > 1:
                    X_train = month_train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                    y_train = month_train_data['wl_up']

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                else:
                    continue  # ถ้าไม่มีข้อมูลทั้งเดือน ข้ามไป

        # ทำนายค่าที่หายไป
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

    # ให้ผู้ใช้เลือกช่วงวันที่และเวลาที่ต้องการตัดข้อมูล
    st.subheader("เลือกช่วงวันที่และเวลาที่ต้องการตัดข้อมูล")
    start_date = st.date_input("เลือกวันเริ่มต้น", pd.to_datetime(filtered_data.index.min()).date())
    start_time = st.time_input("เลือกเวลาเริ่มต้น", value=pd.to_datetime(filtered_data.index.min()).time())
    end_date = st.date_input("เลือกวันสิ้นสุด", pd.to_datetime(filtered_data.index.max()).date())
    end_time = st.time_input("เลือกเวลาสิ้นสุด", value=pd.to_datetime(filtered_data.index.max()).time())

    # รวมวันและเวลาที่เลือกเข้าด้วยกันเป็นช่วงเวลา
    start_datetime = pd.to_datetime(f"{start_date} {start_time}")
    end_datetime = pd.to_datetime(f"{end_date} {end_time}")

    # ตรวจสอบว่ามีข้อมูลในช่วงวันที่และเวลาที่เลือกหรือไม่
    if not filtered_data.index.isin(pd.date_range(start=start_datetime, end=end_datetime)).any():
        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
    else:
        # แสดงกราฟข้อมูลทั้งเดือนก่อนทำการตัด
        st.subheader('กราฟข้อมูลทั้งเดือนก่อนทำการตัด')
        plot_full_month_data(filtered_data, start_datetime)

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












