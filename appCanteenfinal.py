import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# ดึงตัวแปรและฟังก์ชันจากไฟล์ canteen_common.py ของจริง
from canteen_common import (
    TOTAL_TABLES, FLOOD_LABELS_TH, 
    WEEKDAYS_TH, MODEL_FILE, build_input, load_data
)

# --- ตั้งค่าหน้าเพจหลัก ---
st.set_page_config(page_title="Smart Canteen by SekSolo", page_icon="🍽️", layout="wide")

st.title("🍽️ ระบบแนะนำโรงอาหารอัจฉริยะ (Smart Canteen)")
st.markdown("**โรงอาหารหอพักนิสิต (โรงส้ม) มหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน**")
st.sidebar.markdown("### 👨‍💻 พัฒนาโดยกลุ่ม: SekSolo")

# --- โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / MODEL_FILE
    if model_path.exists():
        return joblib.load(model_path)
    return None

model_data = load_model()

# --- สร้าง Tabs ---
tab1, tab2, tab3 = st.tabs(["🎯 ระบบแนะนำเวลา", "📊 ข้อมูลจากการสำรวจจริง", "🧠 ประสิทธิภาพโมเดล AI"])

# ==========================================
# TAB 1: ระบบแนะนำเวลา (Prediction)
# ==========================================
with tab1:
    st.header("🔍 คาดการณ์ความหนาแน่นของโรงอาหาร")
    st.markdown("โปรดระบุข้อมูลช่วงเวลาและสภาพอากาศเพื่อประเมินความหนาแน่นด้วย AI")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # สลับ Key-Value สำหรับแสดงผลภาษาไทย
        day_options = {v: k for k, v in WEEKDAYS_TH.items()}
        selected_day_th = st.selectbox("📅 เลือกวัน", list(day_options.keys()))
        selected_time = st.selectbox("⏰ เลือกเวลา", [f"12:{str(m).zfill(2)}" for m in range(0, 60, 5)])
        
    with col2:
        flood_options_th = {v: k for k, v in FLOOD_LABELS_TH.items()}
        selected_flood_th = st.selectbox("🌧️ สภาพน้ำท่วม/ฝน", list(flood_options_th.keys()))
        
    with col3:
        input_max_temp = st.number_input("📈 อุณหภูมิสูงสุดรายวัน (°C)", value=34.0, step=0.1)
        input_rainfall = st.number_input("💧 ปริมาณน้ำฝนรายวัน (mm)", value=0.0, step=0.1)
    
    if st.button("🚀 ประมวลผลด้วย AI", type="primary"):
        if model_data is None:
            st.error("⚠️ ไม่พบไฟล์ AI (`canteen_model.pkl`) กรุณารัน `train.py` บนคอมพิวเตอร์เพื่อสร้างโมเดลก่อนครับ")
        else:
            model = model_data["model"]
            
            # แปลงค่าภาษาไทยกลับเป็นค่าที่ AI เข้าใจ
            selected_weekday_en = day_options[selected_day_th]
            selected_flood_en = flood_options_th[selected_flood_th]
            
            # สร้างตารางข้อมูล 12 ช่วงเวลา
            input_df = build_input(
                flood_cond=selected_flood_en,
                max_temp=input_max_temp,
                rainfall=input_rainfall,
                weekday=selected_weekday_en
            )
            
            # ให้ AI ทำนาย (ผลลัพธ์เป็น Vacancy Rate)
            predictions_rate = model.predict(input_df)
            
            # คำนวณเป็นจำนวนโต๊ะว่าง (Rate * จำนวนโต๊ะทั้งหมด)
            input_df['Predicted_Vacant'] = np.clip(predictions_rate * TOTAL_TABLES, 0, TOTAL_TABLES).astype(int)
            
            # ดึงเฉพาะเวลาที่ผู้ใช้เลือกมาแสดงผล
            result = input_df[input_df['Time'] == selected_time].iloc[0]
            predicted_vacant = result['Predicted_Vacant']
            
            st.divider()
            st.subheader("ผลการคาดการณ์จาก AI")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("คาดการณ์จำนวนโต๊ะว่าง (Vacant Table)", f"{predicted_vacant} โต๊ะ", f"จากทั้งหมด {TOTAL_TABLES} โต๊ะ")
            
            if predicted_vacant > 15:
                st.success("✅ **คำแนะนำ:** โรงอาหารมีโต๊ะว่างเพียงพอ คุณสามารถไปรับประทานอาหารช่วงเวลานี้ได้เลย!")
            elif predicted_vacant > 5:
                st.warning("⚠️ **คำแนะนำ:** โรงอาหารเริ่มหนาแน่น อาจจะต้องใช้เวลาหาโต๊ะเล็กน้อย")
            else:
                st.error("🚨 **คำแนะนำ:** โรงอาหารมีความหนาแน่นสูงมาก! แนะนำให้หลีกเลี่ยงหรือรออีก 15-20 นาที")

            # แสดงกราฟเส้นแนวโน้มของทุกช่วงเวลาในวันนั้น
            st.markdown("**แนวโน้มโต๊ะว่างในช่วงเวลา 12:00 - 12:55 น.**")
            st.line_chart(input_df.set_index('Time')['Predicted_Vacant'], use_container_width=True)

# ==========================================
# TAB 2: ข้อมูลจากการสำรวจจริง (Dataset)
# ==========================================
with tab2:
    st.header("📂 ข้อมูลที่ใช้ในการฝึกสอนโมเดล (Training Data)")
    try:
        df = load_data()
        st.dataframe(df, use_container_width=True)
        
        st.subheader("📈 จำนวนโต๊ะว่างเฉลี่ยตามช่วงเวลา")
        avg_time = df.groupby('Time')['Vacant'].mean().reset_index()
        fig2 = px.line(avg_time, x='Time', y='Vacant', markers=True, title="โต๊ะว่างเฉลี่ย")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️️ ไม่สามารถโหลดไฟล์ข้อมูล Canteen Data.xlsx ได้: {e}")

# ==========================================
# TAB 3: ประสิทธิภาพโมเดล (Model Performance)
# ==========================================
with tab3:
    st.header("⚙️ สรุปผลการประเมินโมเดล (Model Evaluation)")
    if model_data:
        st.success(f"ใช้โมเดล: **{model_data['model_name']}**")
        st.markdown(f"**ชุดฟีเจอร์ที่ใช้:** {model_data['feature_set']}")
        
        st.subheader("📊 ผลการทดสอบโมเดล (Test Set)")
        st.table(model_data['test_eval'])
        
        st.subheader("📈 เปรียบเทียบทุกโมเดลที่ทดลอง (Model Comparison)")
        st.table(model_data['fs_compare'])
    else:
        st.warning("⚠️️ ยังไม่มีข้อมูลการประเมินโมเดล กรุณารัน train.py ก่อน")