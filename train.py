"""
เทรนโมเดลทำนาย Vacancy Rate (สัดส่วนโต๊ะว่าง) ของโรงอาหารหอพักนิสิต (โรงส้ม) ตามข้อเสนอโครงงาน
ขั้นตอน (ตาม Proposal):
  1) โหลดและทำความสะอาดข้อมูล (144 แถว)
  2) แบ่ง Training Set 120 แถว (10 วัน) / Test Set 24 แถว (2 วัน)
  3) เทรน 3 โมเดลด้วย Training Set: Linear Regression (Baseline), Decision Tree, Random Forest
  4) ประเมินด้วย MAE, MSE และ RMSE บน Test Set
  5) เลือกโมเดลที่ RMSE ต่ำสุดไปใช้ในเว็บแอป และบันทึกเป็น canteen_model.pkl (โมเดลที่เทรนด้วย 120 แถว)
หมายเหตุ: ไม่ใช้ OccupiedTable / VacantTable เป็นฟีเจอร์ เพราะเป็นตัวสร้าง Target (data leakage)
"""
import joblib
import numpy as np
import pandas as pd

from canteen_common import (BASE_DIR, MODEL_FILE, PROPOSAL_FEATURES, PROPOSAL_FS_NAME, TOTAL_TABLES,
                            data_fingerprint, library_versions, load_data, make_models, make_pipeline,
                            metrics, split_by_day)

# ==========================================
# 1. LOAD DATA
# ==========================================
df = load_data()
y = df["VacancyRate"]
print("================================")
print("DATA INFORMATION")
print("================================")
print(f"Observations : {len(df)}")
print(f"Days         : {df['DayNo'].nunique()}")
print(f"Vacancy Rate : min={y.min():.2f}, mean={y.mean():.2f}, max={y.max():.2f}")

# ==========================================
# 2. SPLIT: TRAIN 120 / TEST 24
# ==========================================
num_cols, cat_cols = PROPOSAL_FEATURES
FEATURES = num_cols + cat_cols
test_days, test_mask = split_by_day(df)
X_train, y_train = df.loc[~test_mask, FEATURES], y[~test_mask]
X_test, y_test = df.loc[test_mask, FEATURES], y[test_mask]
print("\n================================")
print(f"TRAIN {len(X_train)} แถว / TEST {len(X_test)} แถว (วันที่ใช้ทดสอบ: {test_days})")
print(f"ฟีเจอร์: {FEATURES}")
print("================================")

# ==========================================
# 3-4. TRAIN 3 MODELS (เทรนด้วย Training Set เท่านั้น) + EVALUATE บน Test Set
# ==========================================
fitted, rows = {}, []
for m_name, model in make_models().items():
    fitted[m_name] = make_pipeline(model, num_cols, cat_cols).fit(X_train, y_train)
    rows.append({"Model": m_name, **metrics(y_test, fitted[m_name].predict(X_test))})
test_eval = pd.DataFrame(rows).set_index("Model")
show = test_eval[["MAE", "MSE", "RMSE"]].copy()
show.insert(1, "MAE (โต๊ะ)", show["MAE"] * TOTAL_TABLES)
print(show.round(4).to_string())

# ==========================================
# 5. เลือกโมเดลที่ดีที่สุด 1 ตัว (RMSE ต่ำสุดบน Test Set) + SAVE
# ==========================================
best_model = test_eval["RMSE"].idxmin()
joblib.dump({"model": fitted[best_model], "model_name": best_model,
             "features": FEATURES, "feature_set": PROPOSAL_FS_NAME,
             "test_mae": float(test_eval.loc[best_model, "MAE"]),
             "test_eval": test_eval, "test_days": test_days,
             "n_train": int(len(X_train)), "n_test": int(len(X_test)),
             "versions": library_versions(),
             "data_fingerprint": data_fingerprint(df), "n_rows": int(len(df)),
             "format": 2},
            BASE_DIR / MODEL_FILE)
print(f"\nโมเดลที่ดีที่สุด: {best_model} (RMSE ต่ำสุด) -> บันทึกเป็น {BASE_DIR / MODEL_FILE}")
print("รันเว็บแอปด้วยคำสั่ง: streamlit run app_v2.py")
