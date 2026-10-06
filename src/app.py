import gradio as gr
import pandas as pd
import joblib

# 1. โหลดโมเดล
artifact = joblib.load("src/insurance_model.joblib")
loaded_model = artifact["model"]

# 2. ฟังก์ชันทำนายผล (แก้ไขการแปลงข้อมูลให้ตรงกับที่โมเดลต้องการเป๊ะๆ)
def predict_insurance_gradio(age, sex, bmi, children, smoker, region):
    
    # แปลงข้อความเป็น 0 และ 1 แบบ Manual (ปลอดภัยกว่าการใช้ get_dummies สำหรับข้อมูล 1 แถว)
    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0
    
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0
    
    # จัดเตรียมข้อมูลทั้งหมดให้อยู่ในรูปแบบ Dictionary
    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": sex_male,
        "smoker_yes": smoker_yes,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest,
        "bmi_smoker": bmi * smoker_yes  # Interaction Term
    }
    
    # แปลงเป็น DataFrame
    df_ready = pd.DataFrame([input_data])
    
    # ใช้ feature_names_in_ ที่อยู่ในโมเดลมาใช้จัดเรียงคอลัมน์โดยอัตโนมัติ
    df_ready = df_ready.reindex(columns=loaded_model.feature_names_in_, fill_value=0)
    
    # ทำนายผล
    prediction = loaded_model.predict(df_ready)[0]
    return round(float(prediction), 2)

# 3. กำหนดค่า Min, Max, Default สำหรับ Slider
age_min, age_max, age_default = 18, 64, 39
bmi_min, bmi_max, bmi_default = 15.0, 53.0, 30.4
children_min, children_max, children_default = 0, 5, 1

# 4. สร้าง UI ด้วย Gradio Interface
app = gr.Interface(
    fn=predict_insurance_gradio,
    inputs=[
        gr.Slider(age_min, age_max, value=age_default, step=1, label="Age (อายุ)"),
        gr.Radio(choices=["female", "male"], value="female", label="Sex (เพศ)"),
        gr.Slider(bmi_min, bmi_max, value=bmi_default, step=0.1, label="BMI (ดัชนีมวลกาย)"),
        gr.Slider(children_min, children_max, value=children_default, step=1, label="Children (จำนวนบุตร)"),
        gr.Radio(choices=["no", "yes"], value="no", label="Smoker (สูบบุหรี่หรือไม่)"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], value="southwest", label="Region (ภูมิภาค)")
    ],
    outputs=gr.Number(label="Predicted Medical Charges ($)"),
    title="Insurance Medical Cost Prediction — MVP",
    description="แอปพลิเคชันพยากรณ์ค่ารักษาพยาบาลใน USA ด้วย Linear Regression"
)

# 5. สั่งรันแอปพลิเคชัน
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")