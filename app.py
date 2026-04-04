import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import time
from datetime import datetime
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Alzheimer Classification System", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f4f6f9;
}

.main-title {
    font-size: 30px;
    font-weight: 600;
    text-align: center;
    color: #2c3e50;
}

.subtitle {
    text-align: center;
    font-size: 14px;
    color: #566573;
    margin-bottom: 30px;
}

.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 6px;
    height: 42px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="main-title">Blood Biomarkers and Brain Imaging Based Alzheimer’s Disease Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Based Clinical and MRI Classification System</p>', unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
clinical_model = joblib.load("clinical_model.pkl")
scaler = joblib.load("clinical_scaler.pkl")
mri_model = joblib.load("mri_model.pkl")
label_encoder = joblib.load("mri_label_encoder.pkl")

# ---------- INPUT ----------
left, right = st.columns(2)

with left:
    st.subheader("Patient Clinical Data")

    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

    age = st.number_input("Age", 40, 100, 60)
    mmse = st.number_input("MMSE Score", 0, 30, 20)
    cdr = st.number_input("CDR Score", 0.0, 3.0, 1.0)
    memory = st.number_input("Memory Score", 0.0, 1.0, 0.5)

    gender = st.selectbox("Gender", ["male", "female"])

with right:
    st.subheader("Brain MRI Image")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, width=350)

st.markdown("---")

# ---------- CLASSIFICATION ----------
if st.button("Run Classification"):

    with st.spinner("Processing..."):
        time.sleep(1)

        # ---------- Clinical Model ----------
        clinical_input = pd.DataFrame(
            [[gender, mmse, age, cdr, memory]],
            columns=["Gender","mmse","ageAtEntry","cdr","memory"]
        )

        clinical_input = pd.get_dummies(clinical_input)

        training_columns = scaler.feature_names_in_

        clinical_input = clinical_input.reindex(columns=training_columns, fill_value=0)

        clinical_scaled = scaler.transform(clinical_input)

        clinical_class = clinical_model.predict(clinical_scaled)[0]
        clinical_conf = np.max(clinical_model.predict_proba(clinical_scaled)) * 100

        # ---------- MRI MODEL ----------
        if uploaded_file is not None:

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

            img = cv2.imdecode(file_bytes,1)

            img = cv2.resize(img,(64,64))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = img.flatten().reshape(1,-1)

            mri_class_encoded = mri_model.predict(img)

            mri_class = label_encoder.inverse_transform(mri_class_encoded)[0]

            mri_conf = np.max(mri_model.predict_proba(img)) * 100

            # ---------- LABEL MAPPING ----------
            label_map = {
                "AD Dementia":"Alzheimer’s Disease Detected",
                "No dementia":"No Alzheimer’s Disease",
                "uncertain dementia":"Possible Cognitive Impairment",
                "Non Demented":"No Alzheimer’s Disease",
                "Very Mild Dementia":"Very Mild Alzheimer’s Stage",
                "Mild Dementia":"Mild Alzheimer’s Stage",
                "Moderate Dementia":"Moderate Alzheimer’s Stage"
            }

            clinical_result = label_map.get(clinical_class, clinical_class)

            mri_result = label_map.get(mri_class, mri_class)

            # ---------- FINAL CLASSIFICATION ----------
            if mri_class == "Non Demented":
                final_result = "No Alzheimer’s Disease Detected"

            elif mri_class == "Very Mild Dementia":
                final_result = "Alzheimer’s Disease Detected – Very Mild Stage"

            elif mri_class == "Mild Dementia":
                final_result = "Alzheimer’s Disease Detected – Mild Stage"

            elif mri_class == "Moderate Dementia":
                final_result = "Alzheimer’s Disease Detected – Moderate Stage"

            else:
                final_result = "Classification Uncertain"

            # ---------- DISPLAY RESULTS ----------
            st.subheader("Classification Results")

            st.write(f"Clinical Model Result: {clinical_result} | Confidence: {clinical_conf:.2f}%")

            st.write(f"MRI Model Result: {mri_result} | Confidence: {mri_conf:.2f}%")

            st.success(f"Final Classification: {final_result}")

            # ---------- PDF REPORT ----------
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            elements = []
            styles = getSampleStyleSheet()

            report_id = f"AD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Title
            elements.append(Paragraph("Alzheimer’s Disease Classification Report", styles["Title"]))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"Report ID: {report_id}", styles["Normal"]))
            elements.append(Paragraph(f"Date: {report_time}", styles["Normal"]))
            elements.append(Spacer(1, 16))

            # ---------- Patient Information ----------
            elements.append(Paragraph("Patient Information", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            patient_data = [
                ["Patient Name", patient_name],
                ["Patient ID", patient_id],
                ["Age", str(age)],
                ["Gender", gender],
                ["MMSE Score", str(mmse)],
                ["CDR Score", str(cdr)],
                ["Memory Score", str(memory)],
            ]

            patient_table = Table(patient_data, colWidths=[180, 320])
            patient_table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ]))
            elements.append(patient_table)
            elements.append(Spacer(1, 16))

            # ---------- Model Results ----------
            elements.append(Paragraph("Model Results", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            results_data = [
                ["Model", "Result", "Confidence"],
                ["Clinical Model", clinical_result, f"{clinical_conf:.2f}%"],
                ["MRI Model", mri_result, f"{mri_conf:.2f}%"],
                ["Final Classification", final_result, "-"],
            ]

            results_table = Table(results_data, colWidths=[160, 240, 100])
            results_table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ]))
            elements.append(results_table)
            elements.append(Spacer(1, 16))

            # ---------- Result Interpretation ----------
            elements.append(Paragraph("Result Interpretation", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            elements.append(Paragraph(
                f"The clinical biomarker model classified the case as: <b>{clinical_result}</b>. "
                f"The MRI imaging model classified the brain scan as: <b>{mri_result}</b>. "
                f"Based on the system fusion rule, the final classification result is: "
                f"<b>{final_result}</b>.",
                styles["Normal"]
            ))
            elements.append(Spacer(1, 16))

            # ---------- Clinical Score Guide ----------
            elements.append(Paragraph("Clinical Score Interpretation", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            elements.append(Paragraph(
                "MMSE Score Guide: 24–30 Normal cognition | 18–23 Mild cognitive impairment | 0–17 Severe impairment.",
                styles["Normal"]
            ))

            elements.append(Paragraph(
                "CDR Score Guide: 0 Normal | 0.5 Very Mild Dementia | 1 Mild Dementia | 2 Moderate Dementia | 3 Severe Dementia.",
                styles["Normal"]
            ))
            elements.append(Spacer(1, 16))

            

            # ---------- Recommendations ----------
            elements.append(Paragraph("General Recommendations", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            elements.append(Paragraph("• Maintain regular cognitive health checkups.", styles["Normal"]))
            elements.append(Paragraph("• Follow balanced nutrition and physical activity.", styles["Normal"]))
            elements.append(Paragraph("• Engage in mental activities such as reading or puzzles.", styles["Normal"]))
            elements.append(Paragraph("• Consult a neurologist for professional medical evaluation if symptoms persist.", styles["Normal"]))
            elements.append(Spacer(1, 16))

            # ---------- Disclaimer ----------
            elements.append(Paragraph("Disclaimer", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            elements.append(Paragraph(
                "This report is generated by an AI-based classification system for academic research purposes. "
                "It does not constitute a medical diagnosis and should not replace professional clinical evaluation.",
                styles["Normal"]
            ))
            elements.append(Spacer(1, 30))

            elements.append(Paragraph("Authorized Signature: ____________________", styles["Normal"]))

            doc.build(elements)
            pdf = buffer.getvalue()
            buffer.close()

            st.download_button(
                label="Download Classification Report (PDF)",
                data=pdf,
                file_name="Alzheimer_Classification_Report.pdf",
                mime="application/pdf"
            )

        else:
         st.warning("Please upload MRI image.")