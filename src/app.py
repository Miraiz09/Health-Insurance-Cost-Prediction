import gradio as gr
import pandas as pd
import joblib

# 1. โหลด Pipeline ที่เทรนไว้ (รับตัวหนังสือตรงๆ ได้เลย)
model = joblib.load('src/insurance_model.joblib')

# 2. ฟังก์ชันทำนายผล
def predict_charges(age, sex, bmi, children, smoker, region):
    # นำ Input จากหน้าเว็บมาจัดเรียงเป็น DataFrame ให้คอลัมน์ตรงกับตอนเทรน
    input_data = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    )
    
    # สั่งทำนายผล (Pipeline จะจัดการแปลงข้อความให้เอง)
    prediction = model.predict(input_data)[0]
    
    # จัด Format ตัวเลขเป็นสกุลเงิน
    return f"${prediction:,.2f}"

# 3. สร้าง UI ของ Gradio
interface = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Slider(minimum=18, maximum=100, value=25, step=1, label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(minimum=15.0, maximum=55.0, value=25.0, step=0.1, label="BMI"),
        gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
    ],
    outputs=gr.Textbox(label="Predicted Medical Charges"),
    title="Health Insurance Cost Predictor",
    description="ใส่ข้อมูลสุขภาพเพื่อประเมินค่าใช้จ่ายประกันสุขภาพเบื้องต้น"
)

# 4. เปิดใช้งานหน้าเว็บ
if __name__ == "__main__":
    interface.launch()