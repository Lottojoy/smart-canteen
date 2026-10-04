"""ฟังก์ชันที่ใช้ร่วมกันระหว่าง train.py และ app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

TOTAL_TABLES = 196
RANDOM_STATE = 42
MODEL_FILE = "canteen_model.pkl"
BASE_DIR = Path(__file__).resolve().parent

# ลำดับคอลัมน์ในไฟล์ -> ชื่อสั้น
COLUMN_MAP = {
    "Day(1-15)": "DayNo",
    "Time": "Time",
    "Day(Monday - Friday)": "Weekday",
    "Vacant Table": "Vacant",
    "OccupiedTable": "Occupied",
    "Kamphaeng Saen Temperature Condition": "TempCond",
    "Rain and Flood Condition(Nakhon Pathom)": "FloodCond",
    "Daily Maximum Temperature(Nakhon Pathom)": "MaxTemp",
    "Daily Rainfall (mm) (Nakhon Pathom)": "Rainfall",
}

FLOOD_OPTIONS = ["No Flooding", "Flooding", "Continuous Rain"]
FLOOD_LABELS_TH = {"No Flooding": "ปกติ (ไม่มีน้ำท่วม)", "Flooding": "น้ำท่วม",
                   "Continuous Rain": "ฝนตกต่อเนื่อง"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
WEEKDAYS_TH = {"monday": "จันทร์", "tuesday": "อังคาร", "wednesday": "พุธ",
               "thursday": "พฤหัสบดี", "friday": "ศุกร์"}
HOT_THRESHOLD = 34.0   # อุณหภูมิสูงสุด >= 34°C = ร้อน (ตรงกับข้อมูลจริง: วันที่ 12 อุณหภูมิ 34.0 บันทึกว่า Hot)

# ชุดฟีเจอร์ที่จะเปรียบเทียบ: (คอลัมน์ตัวเลข, คอลัมน์หมวดหมู่)
FEATURE_SETS = {
    "เวลาอย่างเดียว":               (["Minute", "Minute2"], []),
    "เวลา + วันในสัปดาห์":           (["Minute", "Minute2"], ["Weekday"]),
    "เวลา + สภาพอากาศแย่":          (["Minute", "Minute2", "BadWeather"], []),
    "เวลา + สภาพอากาศแย่ + อุณหภูมิ": (["Minute", "Minute2", "BadWeather", "MaxTemp"], []),
    "เวลา + สภาพอากาศทั้งหมด":      (["Minute", "Minute2", "MaxTemp", "Rainfall", "IsHot"], ["FloodCond"]),
    "ทุกอย่าง (รวมวัน)":            (["Minute", "Minute2", "MaxTemp", "Rainfall", "IsHot"], ["FloodCond", "Weekday"]),
}


# ฟีเจอร์ที่ใช้เทรนจริง ยึดตาม Input Data ใน Proposal: วันในสัปดาห์, เวลา, สภาพอุณหภูมิ, สภาพฝนและน้ำท่วม,
# อุณหภูมิสูงสุด, ปริมาณน้ำฝน (ไม่ใช้จำนวนโต๊ะว่าง/โต๊ะมีคนนั่ง เพราะเป็นตัวสร้าง Target = data leakage)
# BadWeather = จัดกลุ่ม "น้ำท่วม" และ "ฝนตกต่อเนื่อง" รวมกัน (ฝนตกต่อเนื่องมีแค่ 1 วัน เรียนรู้แยกไม่ได้)
PROPOSAL_FEATURES = (["Minute", "Minute2", "MaxTemp", "Rainfall", "BadWeather"], ["Weekday", "TempCond"])
PROPOSAL_FS_NAME = "ตาม Proposal (วัน เวลา อุณหภูมิ ฝน/น้ำท่วม)"


def find_data_file():
    """หาไฟล์ข้อมูลในโฟลเดอร์เดียวกับสคริปต์ (รองรับหลายชื่อ)"""
    for name in ["Canteen_Data.xlsx", "Canteen Data.xlsx"]:
        if (BASE_DIR / name).exists():
            return BASE_DIR / name
    raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล .xlsx ในโฟลเดอร์ {BASE_DIR} "
                            "ให้วาง Canteen_Data.xlsx ไว้ข้าง train.py")


def add_derived_features(df):
    """สร้างคอลัมน์ Minute, Minute2, BadWeather, IsHot จากคอลัมน์ดิบ"""
    t = pd.to_datetime(df["Time"].astype(str).str.strip().str[:5], format="%H:%M")
    df["Minute"] = (t.dt.hour * 60 + t.dt.minute) - 12 * 60   # นาทีนับจาก 12:00
    df["Minute2"] = df["Minute"] ** 2                         # ให้จับกราฟโค้งได้
    df["BadWeather"] = df["FloodCond"].isin(["Flooding", "Continuous Rain"]).astype(int)
    df["IsHot"] = (df["MaxTemp"] >= HOT_THRESHOLD).astype(float)
    df.loc[df["MaxTemp"].isna(), "IsHot"] = np.nan
    return df


def load_data():
    """โหลด + ทำความสะอาดข้อมูล -> DataFrame พร้อมใช้ (มีคอลัมน์ VacancyRate)"""
    path = find_data_file()
    df = pd.read_excel(path, header=1)            # แถวแรกของไฟล์เป็นแถวขยะ
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())].copy()

    for c in ["DayNo", "Vacant", "Occupied", "MaxTemp", "Rainfall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DayNo", "Vacant", "Occupied"]).copy()   # ตัดแถวหมายเหตุท้ายไฟล์

    df["Weekday"] = df["Weekday"].astype(str).str.strip().str.lower()
    df["FloodCond"] = df["FloodCond"].astype(str).str.strip()

    # แถวผิดปกติ: โต๊ะว่าง + โต๊ะมีคนนั่ง ต้องได้ 196
    bad = (df["Vacant"] + df["Occupied"]) != TOTAL_TABLES
    if bad.any():
        print("ตัดแถวที่ผลรวมโต๊ะไม่เท่ากับ 196 (ตรวจสอบกับใบบันทึกต้นฉบับ):")
        print(df.loc[bad, ["DayNo", "Time", "Vacant", "Occupied"]].to_string(index=False), "\n")
        df = df[~bad].copy()

    df = add_derived_features(df)
    df["VacancyRate"] = df["Vacant"] / TOTAL_TABLES      # Target Variable
    return df.reset_index(drop=True)


def build_input(flood_cond="No Flooding", max_temp=33.0, rainfall=0.0, weekday="monday"):
    """สร้างตารางอินพุต 12 ช่วงเวลา (12:00-12:55) จากสภาพอากาศที่ผู้ใช้เลือก"""
    minutes = np.arange(0, 60, 5)
    d = pd.DataFrame({"Minute": minutes})
    d["Minute2"] = d["Minute"] ** 2
    d["FloodCond"] = flood_cond
    d["BadWeather"] = int(flood_cond in ("Flooding", "Continuous Rain"))
    d["MaxTemp"] = max_temp
    d["Rainfall"] = rainfall
    d["IsHot"] = float(max_temp >= HOT_THRESHOLD)
    d["TempCond"] = "Hot" if max_temp >= HOT_THRESHOLD else "Not Hot"
    d["Weekday"] = weekday
    d["Time"] = [f"12:{m:02d}" for m in minutes]
    return d


# ---------- ส่วนที่ใช้ร่วมกันระหว่าง train.py และ compare_models.py ----------
def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5,
                                               random_state=RANDOM_STATE),
    }


def make_pipeline(model, num_cols, cat_cols):
    """Pipeline เดียวกันทั้ง train.py และ compare_models.py (รองรับทั้งคอลัมน์ตัวเลขและหมวดหมู่)"""
    parts = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", model)])


def metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 1)
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}


def library_versions():
    """เวอร์ชันไลบรารีที่ใช้เทรน เก็บไว้ใน .pkl เพื่อเตือนถ้าเวอร์ชันตอนรันแอปไม่ตรง"""
    import sklearn
    return {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def data_fingerprint(df):
    """ลายนิ้วมือของข้อมูลที่ผ่านการทำความสะอาดแล้ว (ไม่เปลี่ยนเมื่อแค่กด save ไฟล์ Excel โดยไม่แก้ข้อมูล)
    train.py เก็บค่านี้ไว้ใน .pkl และแอปใช้เทียบ เพื่อเตือนเมื่อแก้ข้อมูลแต่ยังไม่ได้เทรนใหม่"""
    import hashlib
    cols = ["DayNo", "Time", "Weekday", "Vacant", "Occupied", "FloodCond", "MaxTemp", "Rainfall"]
    text = df[cols].astype(str).to_csv(index=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def split_by_day(df):
    """แบ่ง Train/Test ตามวัน (ข้อมูลที่เก็บห่างกัน 5 นาทีคล้ายกันมาก การสุ่มรายแถวจะทำให้ข้อมูลรั่วเข้าชุดทดสอบ)
    วันทดสอบ 2 วัน = วันปกติ 1 วัน + วันสภาพอากาศแย่ 1 วัน (random seed คงที่) -> Train 120 / Test 24 แถว
    คืนค่า (test_days, test_mask)"""
    rng = np.random.RandomState(RANDOM_STATE)
    day_bad = df.groupby("DayNo")["BadWeather"].max()
    test_days = sorted(int(d) for d in [rng.choice(day_bad[day_bad == 0].index.to_numpy()),
                                        rng.choice(day_bad[day_bad == 1].index.to_numpy())])
    return test_days, df["DayNo"].isin(test_days).to_numpy()
