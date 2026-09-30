# 🏥 Medical Insurance Cost Prediction
> **รายวิชา:** 03603351 Introduction to Data Science  
> **เทคโนโลยี:** Python | Machine Learning (Linear Regression) | Gradio

โปรเจกต์การประมวลผลและวิเคราะห์ข้อมูลเพื่อสร้างโมเดล Machine Learning ในการทำนายค่าเบี้ยประกันสุขภาพ (Medical Insurance Charges) ในสหรัฐอเมริกา โดยอ้างอิงจากข้อมูลปัจจัยทางประวัติสุขภาพและพฤติกรรมส่วนบุคคล

---

**1.** 🎯 เป้าหมายและผลลัพธ์ (Objective & Key Results)

* **เป้าหมาย:** สร้างโมเดลทำนายค่าเบี้ยประกันสุขภาพที่มีความแม่นยำและช่วยประเมินความเสี่ยงได้อย่างสมเหตุสมผล
* **ผลลัพธ์ที่ได้ (Model Performance):**
  * **R² Score:** `86.5%` (โมเดลสามารถอธิบายความแปรผันของค่าประกันได้ถึง 86.5%)
  * **Mean Absolute Error (MAE):** `$2,756.90` (คลาดเคลื่อนเฉลี่ยประมาณ 21% จากค่าเบี้ยประกันเฉลี่ยหลักที่ ≈ $13,000.00)

---

**2.** 📊 ชุดข้อมูล (Dataset)

* **แหล่งข้อมูล:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* **ขนาดข้อมูล:** 1,338 แถว, 7 คอลัมน์
* **โครงสร้างข้อมูล (Features):**
  1. `age` : อายุของผู้ประกันตน (int64)
  2. `sex` : เพศ (`female`, `male`) (object)
  3. `bmi` : ดัชนีมวลกาย Body Mass Index (float64)
  4. `children` : จำนวนบุตร/ผู้อยู่ในอุปการะ (int64)
  5. `smoker` : ประวัติการสูบบุหรี่ (`yes`, `no`) (object)
  6. `region` : ภูมิภาคที่อยู่อาศัยใน US (`northeast`, `northwest`, `southeast`, `southwest`) (object)
  7. `charges` : **[Target]** ค่าเบี้ยประกันสุขภาพที่ต้องจ่าย (float64)

---

**3.** 🔍 ขั้นตอนการดำเนินงาน (Data Pipeline & Analysis)

1. **Data Cleaning & Quality Check:**
   * **Data Type:** ตรวจสอบความถูกต้องของประเภทข้อมูล
   * **Missing Values:** ไม่พบค่าสูญหาย (Null Values = 0)
   * **Duplicates:** พบข้อมูลซ้ำกันสมบูรณ์ 1 รายการ (พิจารณาว่าเป็นไปได้เนื่องจากไม่มี Primary Key/ID)
   * **Outlier Detection:** ตรวจสอบด้วยวิธี IQR พบ Outliers ใน `bmi` (0.67%) และ `charges` (10.39%) ซึ่งยังเป็นไปได้ตามความเป็นจริงทางธุรกิจ
   * **Business Logic Check:** ค่า `age`, `bmi`, `children`, `charges` ไม่มีค่าติดลบ และค่าแจงนับ (Categorical values) ถูกต้องตรงตามเงื่อนไข

2. **Confirmatory Data Analysis (CDA):**
   * ตั้งสมมติฐานและตรวจสอบความสัมพันธ์ระหว่างตัวแปรต้นกับค่าเบี้ยประกัน (`charges`) ผ่านการ Plot กราฟ (Barplot & Scatterplot):
     * 🚬 **Smoker:** การสูบบุหรี่ส่งผลกระทบต่อค่าประกันสูงที่สุดอย่างมีนัยสำคัญ
     * 📈 **Age & BMI:** ค่าประกันมีแนวโน้มสูงขึ้นตามอายุและดัชนีมวลกาย (BMI)
     * 🚻 **Sex, Children & Region:** ส่งผลต่อค่าเบี้ยประกันน้อยมากอย่างไม่มีนัยสำคัญ

3. **Model Development & Deployment:**
   * พัฒนาโมเดลด้วย **Linear Regression**
   * บันทึกโมเดลด้วย `joblib`
   * พัฒนา Interactive Web UI สำหรับให้ผู้ใช้ทดลองทำนายค่าเบี้ยประกันผ่าน **Gradio**

---
**4.** 🛠️ การติดตั้งและการใช้งาน (Installation & Setup)

### 1. Clone Repository
https://github.com/Miraiz09/Linear-regression.git

**5.** ข้อมูลผู้พัฒนา
  1.นายชโยดม ชัยรัตน์ (รหัสนิสิต: 6730300124)
  2.นายทรงวุฒิ โคนัก (รหัสนิสิต: 6730300175)
  3.พงศกร เกศนาคินทร์ (รหัสนิสิต: 6730300345)
  4.นายพรหมพิริยะ คำยนต์ (รหัสนิสิต: 6730300388)
