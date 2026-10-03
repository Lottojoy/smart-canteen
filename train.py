"""
เทรนโมเดลทำนาย Vacancy Rate (สัดส่วนโต๊ะว่าง) ของโรงอาหารหอพักนิสิต (โรงส้ม)
ขั้นตอน:
  1) โหลดและทำความสะอาดข้อมูล
  2) เปรียบเทียบชุดฟีเจอร์ด้วย Leave-One-Day-Out (ทดสอบทีละวัน)
  3) เลือกชุดฟีเจอร์ที่ดีที่สุด แล้วประเมินแบบ Train 10 วัน / Test 2 วัน (ประมาณ 120 / 24 แถว ตามข้อเสนอโครงงาน)
  4) เทรนโมเดลสุดท้ายด้วยข้อมูลทั้งหมดและบันทึกเป็น canteen_model.pkl
หมายเหตุ: ไม่ใช้ OccupiedTable / VacantTable เป็นฟีเจอร์ เพราะเป็นตัวสร้าง Target (data leakage)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_error

from canteen_common import (BASE_DIR, FEATURE_SETS, MODEL_FILE, RANDOM_STATE, library_versions,
                            load_data, make_models, make_pipeline, metrics)

N_TEST_DAYS = 2        # 2 วัน x 12 ช่วงเวลา = 24 แถว

# ==========================================
# 1. LOAD DATA
# ==========================================
df = load_data()
y = df["VacancyRate"]
groups = df["DayNo"]
print("================================")
print("DATA INFORMATION")
print("================================")
print(f"Observations : {len(df)}")
print(f"Days         : {groups.nunique()}")
print(f"Vacancy Rate : min={y.min():.2f}, mean={y.mean():.2f}, max={y.max():.2f}")


# ==========================================
# 3. COMPARE FEATURE SETS (Leave-One-Day-Out)
# ==========================================
print("\n================================")
print("เปรียบเทียบชุดฟีเจอร์ (Leave-One-Day-Out, MAE ต่ำ = ดี)")
print("================================")
rows = []
for fs_name, (num_cols, cat_cols) in FEATURE_SETS.items():
    row = {"ชุดฟีเจอร์": fs_name}
    for m_name, model in make_models().items():
        pred = cross_val_predict(make_pipeline(model, num_cols, cat_cols),
                                 df[num_cols + cat_cols], y, groups=groups,
                                 cv=LeaveOneGroupOut())
        row[m_name] = metrics(y, pred)["MAE"]
    row["เฉลี่ย"] = np.mean([row[m] for m in make_models()])
    rows.append(row)
cmp_df = pd.DataFrame(rows).set_index("ชุดฟีเจอร์")
print(cmp_df.round(3).to_string())
baseline = mean_absolute_error(y, np.full(len(y), y.mean()))
print(f"\nBaseline (ทายค่าเฉลี่ยเสมอ) MAE = {baseline:.3f}")

best_fs = cmp_df["เฉลี่ย"].idxmin()
num_cols, cat_cols = FEATURE_SETS[best_fs]
FEATURES = num_cols + cat_cols
print(f"\nชุดฟีเจอร์ที่เลือก: {best_fs} -> {FEATURES}")

# ==========================================
# 4. TRAIN 10 DAYS / TEST 2 DAYS (ตามข้อเสนอ 120 / 24)
# ==========================================
# เลือกวันทดสอบ 2 วัน: วันปกติ 1 วัน + วันสภาพอากาศแย่ 1 วัน (กันไม่ให้ชุดทดสอบมีแต่วันแบบเดียว)
rng = np.random.RandomState(RANDOM_STATE)
day_bad = df.groupby("DayNo")["BadWeather"].max()
normal_days, bad_days = day_bad[day_bad == 0].index.to_numpy(), day_bad[day_bad == 1].index.to_numpy()
test_day_set = [rng.choice(normal_days), rng.choice(bad_days)]
test_mask = groups.isin(test_day_set).to_numpy()
train_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
X_train, X_test = df.iloc[train_idx][FEATURES], df.iloc[test_idx][FEATURES]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_days = sorted(int(d) for d in test_day_set)
print("\n================================")
print(f"TRAIN {len(X_train)} แถว / TEST {len(X_test)} แถว (วันที่ใช้ทดสอบ: {test_days})")
print("================================")

result_rows = []
for m_name, model in make_models().items():
    pipe = make_pipeline(model, num_cols, cat_cols).fit(X_train, y_train)
    result_rows.append({"Model": m_name, **metrics(y_test, pipe.predict(X_test))})
test_eval = pd.DataFrame(result_rows).set_index("Model")
print(test_eval.round(4).to_string())

# ผล Leave-One-Day-Out ของชุดฟีเจอร์ที่เลือก (ครบทุกตัวชี้วัด) ไว้แสดงในเว็บ
lodo_rows = []
for m_name, model in make_models().items():
    pred = cross_val_predict(make_pipeline(model, num_cols, cat_cols), df[FEATURES], y,
                             groups=groups, cv=LeaveOneGroupOut())
    lodo_rows.append({"Model": m_name, **metrics(y, pred)})
lodo_eval = pd.DataFrame(lodo_rows).set_index("Model")
print("(ผลชุด Test มีแค่ 2 วัน ขึ้นกับว่าสุ่มได้วันไหน ให้ดูผล Leave-One-Day-Out ประกอบ)")

# ==========================================
# 5. เลือกโมเดลที่ดีที่สุด 1 ตัว + เทรนด้วยข้อมูลทั้งหมด + SAVE
# ==========================================
# เกณฑ์: RMSE ต่ำสุดจาก Leave-One-Day-Out (ลงโทษความผิดพลาดใหญ่ ซึ่งสำคัญต่อการแนะนำเวลา)
# หมายเหตุ: Decision Tree กับ Random Forest มี MAE เกือบเท่ากัน จึงใช้ RMSE ตัดสิน
best_model = lodo_eval["RMSE"].idxmin()
final_model = make_pipeline(make_models()[best_model], num_cols, cat_cols).fit(df[FEATURES], y)

joblib.dump({"model": final_model, "model_name": best_model,
             "features": FEATURES, "feature_set": best_fs,
             "cv_mae": float(lodo_eval.loc[best_model, "MAE"]),
             "lodo_eval": lodo_eval, "test_eval": test_eval, "fs_compare": cmp_df,
             "baseline_mae": float(baseline), "test_days": test_days,
             "n_train": int(len(X_train)), "n_test": int(len(X_test)),
             "versions": library_versions()},
            BASE_DIR / MODEL_FILE)
print(f"\nโมเดลที่ดีที่สุด: {best_model} (RMSE ต่ำสุด) -> บันทึกเป็น {BASE_DIR / MODEL_FILE}")
print("รันเว็บแอปด้วยคำสั่ง: streamlit run app.py")