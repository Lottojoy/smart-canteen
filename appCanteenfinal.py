import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from PIL import Image
# ดึงตัวแปรและฟังก์ชันจากไฟล์ canteen_common.py
from canteen_common import (
    TOTAL_TABLES, FLOOD_LABELS_TH, 
    WEEKDAYS_TH, MODEL_FILE, build_input, load_data
)

# --- 1. ตั้งค่าหน้าเพจหลัก ---
st.set_page_config(page_title="Smart Canteen by SekSolo", page_icon="🍽️", layout="wide")

# --- CSS Animation (ใส่ Transition แบบนุ่มนวล) ---
st.markdown("""
    <style>
        /* เอฟเฟกต์ Fade-in สำหรับเนื้อหาทั้งหมด */
        .stApp {
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* ปรับให้ Expander (กล่องซ่อนข้อมูล) ค่อยๆ กางออกแบบสมูท */
        .streamlit-expanderContent {
            transition: all 0.3s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

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

# --- 2. สร้าง Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 ระบบแนะนำเวลา", 
    "📊 ข้อมูลจากการสำรวจจริง", 
    "🧠 ประสิทธิภาพโมเดล AI", 
    "ℹ️ เกี่ยวกับโปรเจกต์"
])

# ==========================================
# TAB 1: ระบบแนะนำเวลา (Prediction)
# ==========================================
with tab1:
    st.header("🔍 คาดการณ์ความหนาแน่นของโรงอาหาร")
    st.markdown("โปรดระบุข้อมูลช่วงเวลาและสภาพอากาศเพื่อประเมินความหนาแน่นด้วย AI")
    
    col1, col2, col3 = st.columns(3)
    with col1:
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
            st.error("⚠️ ไม่พบไฟล์ AI (`canteen_model.pkl`) กรุณารัน `train.py` เพื่อสร้างโมเดลก่อน")
        else:
            model = model_data["model"]
            selected_weekday_en = day_options[selected_day_th]
            selected_flood_en = flood_options_th[selected_flood_th]
            
            input_df = build_input(
                flood_cond=selected_flood_en,
                max_temp=input_max_temp,
                rainfall=input_rainfall,
                weekday=selected_weekday_en
            )
            
            predictions_rate = model.predict(input_df)
            input_df['Predicted_Vacant'] = np.clip(predictions_rate * TOTAL_TABLES, 0, TOTAL_TABLES).astype(int)
            
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
        fig2 = px.line(avg_time, x='Time', y='Vacant', markers=True, title="โต๊ะว่างเฉลี่ยในช่วงเที่ยง")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"⚠ ไม่สามารถโหลดไฟล์ข้อมูล Canteen Data.xlsx ได้: {e}")

# ==========================================
# TAB 3: ประสิทธิภาพโมเดล (Model Performance)
# ==========================================
with tab3:
    st.header("⚙️ สรุปผลการประเมินโมเดล (Model Evaluation)")
    if model_data:
        st.success(f"✅ เลือกใช้งานโมเดล: **{model_data['model_name']}** (ชุดฟีเจอร์: {model_data['feature_set']})")
        
        eval_df = model_data['test_eval'].reset_index()
        
        fig_model = px.bar(
            eval_df, 
            x='Model', 
            y='RMSE', 
            color='Model',
            text_auto='.4f',
            title="📊 เปรียบเทียบค่าความคลาดเคลื่อน (RMSE) ของแต่ละโมเดล (ยิ่งน้อยยิ่งดี)",
            labels={'RMSE': 'ค่าความคลาดเคลื่อน RMSE (โต๊ะ)'}
        )
        fig_model.update_layout(showlegend=False)
        st.plotly_chart(fig_model, use_container_width=True)
        
        st.markdown("### 🤔 ทำไม Random Forest ถึงแม่นยำที่สุด?")
        col_reason1, col_reason2, col_reason3 = st.columns(3)
        with col_reason1:
            st.info("**Linear Regression**\n\nมองความสัมพันธ์แบบเส้นตรง ซึ่งในชีวิตจริง คนไม่ได้เข้าโรงอาหารเป็นเส้นตรง (เช่น 12.15 คนจะพุ่งสูงปรี๊ด) โมเดลนี้จึงทายคลาดเคลื่อนเยอะสุด")
        with col_reason2:
            st.warning("**Decision Tree**\n\nใช้การสร้างกฎ (If-Else) ทำให้จับทิศทางเวลาและสภาพอากาศได้ดีกว่า แต่มีข้อเสียคืออาจจะ 'จำข้อสอบ' (Overfitting) มากเกินไป")
        with col_reason3:
            st.success("**Random Forest**\n\nใช้ต้นไม้หลายๆ ต้นมาโหวตกัน (Ensemble) ทำให้ลดความผิดพลาดจากข้อมูลที่แกว่งไปมาได้ดีที่สุด และทนทานต่อปัจจัยแทรกซ้อนอย่างฝนตก")

        with st.expander("📄 ดูตารางข้อมูลเชิงลึก (Raw Data)"):
            st.markdown("**1. ผลการทดสอบโมเดล (Test Set)**")
            st.dataframe(eval_df, use_container_width=True)
            
            st.markdown("**2. เปรียบเทียบทุกชุดฟีเจอร์ที่ทดลอง (Model Comparison)**")
            st.dataframe(model_data['fs_compare'], use_container_width=True)
    else:
        st.warning("⚠ ยังไม่มีข้อมูลการประเมินโมเดล กรุณารัน train.py ก่อน")

# ==========================================
# TAB 4: เกี่ยวกับโปรเจกต์ (About)
# ==========================================
with tab4:
    st.header("📸 ข้อมูลโครงงานและผู้จัดทำ")

    st.info("""
    **📌 Pain Point**  
    ช่วงพักเที่ยง โรงอาหารหอพักนิสิต (โรงส้ม) มักแออัดและโต๊ะไม่พอ ทำให้นิสิตเสียเวลารอ 
    คณะผู้จัดทำจึงพัฒนาระบบ AI คาดการณ์ความหนาแน่นจากปัจจัยด้านเวลาและสภาพอากาศ 
    เพื่อช่วยแนะนำช่วงเวลาเข้าใช้บริการ ให้นิสิตวางแผนได้ง่ายขึ้น ลดการรอคอย และเพิ่มความสะดวกสบาย
    """)
    
    st.subheader("📍 สถานที่ศึกษา: โรงอาหารหอพักนิสิต (โรงส้ม)")
    
    # ฟังก์ชันสำหรับ Crop รูปภาพให้เป็นสี่เหลี่ยมจัตุรัส (1:1) จากจุดกึ่งกลาง
    def crop_to_square(image):
        width, height = image.size
        # หาขนาดด้านที่สั้นที่สุด
        new_size = min(width, height)
        
        # คำนวณพิกัดเพื่อตัดจากตรงกลาง
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        
        return image.crop((left, top, right, bottom))

    # --- แกลลอรีรูปภาพ 3x3 (บีบรูปให้เล็กลงโดยเพิ่มพื้นที่ว่างซ้าย-ขวา) ---
    margin_left, col_img1, col_img2, col_img3, margin_right = st.columns([1.5, 2, 2, 2, 1.5])
    img_cols = [col_img1, col_img2, col_img3]
    
    found_any_image = False
    
    for i in range(1, 10):
        img_name = f"canteen_img_{i}.jpg"
        if os.path.exists(img_name):
            found_any_image = True
            with img_cols[(i - 1) % 3]:
                # เปิดรูปภาพ -> นำไป Crop เป็นสี่เหลี่ยมจัตุรัส -> แสดงผล
                img = Image.open(img_name)
                cropped_img = crop_to_square(img)
                st.image(cropped_img, use_container_width=True)
                
    if not found_any_image:
        st.info("💡 **Tips:** อัปโหลดภาพบรรยากาศโรงอาหาร ตั้งชื่อไฟล์ว่า `canteen_img_1.jpg` ถึง `canteen_img_9.jpg` ลงใน GitHub เพื่อแสดงผลเป็นแกลลอรีตรงนี้")
    
    st.divider()
    
    col_about1, col_about2 = st.columns(2)
    with col_about1:
        st.subheader("📋 หลักฐานการเก็บข้อมูล")
        st.markdown("""
        * **ระยะเวลาการเก็บข้อมูล:** 12 วันทำการ (จันทร์-ศุกร์)
        * **ช่วงเวลา:** 12:00 น. - 13:00 น. (ความถี่ทุก 5 นาที)
        * **วิธีการ:** ลงพื้นที่สังเกตการณ์ นับโต๊ะว่าง และบันทึกสภาพอากาศจริงควบคู่กับฐานข้อมูลจากกรมอุตุนิยมวิทยา
        * **จำนวนข้อมูลรวม:** 120 Observations (ใช้งานจริงหลังจากล้างข้อมูล)
        """)
        
    with col_about2:
        st.subheader("👨‍💻 คณะผู้จัดทำ (ทีม SekSolo)")
        st.markdown("""
        **รายวิชา Artificial Intelligence**
        
        1. นาย ชินณพัฒน์ สีตะสิทธิ์ 
        2. นางสาว พัชริดา หาคูณ 
        3. นาย ศิวกร จันทร์พรม 
        4. นาย ธัชชัย ผาสุข
        5. ชยพล เรือนเพชร
        """)
