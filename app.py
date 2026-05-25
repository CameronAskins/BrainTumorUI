import streamlit as st
from PIL import Image
import pandas as pd
from pipeline import predict_pipeline

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.title-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #101828, #1d3557);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.result-card {
    padding: 22px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.metric-card {
    padding: 15px;
    border-radius: 14px;
    background-color: #ffffff;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="title-box">
    <h1>🧠 Brain Tumor Detection & Classification</h1>
    <p>Two-stage AI pipeline using MRI images for tumor screening and classification</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
**Medical Image Analysis**

**Pipeline**
1. Binary PyTorch model  
   - Tumor vs No Tumor  
2. Multi-class VGG16 model  
   - Glioma  
   - Meningioma  
   - Pituitary  

**Team Members**
- Cameron Askins  
- Axel Espinosa-Chan  
- Williams Okoye  
- Tirth Patel  
""")

st.sidebar.warning("This tool is for educational purposes only and is not a medical diagnosis system.")

# ---------- Main Layout ----------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    else:
        st.info("Upload an MRI image to begin analysis.")

with right_col:
    st.subheader("Prediction Results")

    if uploaded_file is not None:
        with st.spinner("Analyzing MRI image..."):
            result = predict_pipeline(image)

        prediction = result["prediction"]
        confidence = result["confidence"]
        stage = result["stage"]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if "No Tumor" in prediction:
            st.success("No Tumor Detected")
        elif "Uncertain" in prediction:
            st.warning("Uncertain Prediction — Doctor Review Recommended")
        else:
            st.error("Tumor Detected")

        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
        st.markdown(f"**Stage:** {stage}")

        st.markdown('</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        with col_b:
            if "No Tumor" in prediction:
                st.metric("Risk Level", "Low")
            elif "Uncertain" in prediction:
                st.metric("Risk Level", "Review")
            else:
                st.metric("Risk Level", "High")

# ---------- Probability Chart ----------
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Probability Distribution")

    if "all_probabilities" in result and "class_labels" in result:
        chart_df = pd.DataFrame({
            "Class": result["class_labels"],
            "Probability": result["all_probabilities"]
        })

        st.bar_chart(chart_df, x="Class", y="Probability")

        st.dataframe(
            chart_df.assign(
                Probability=chart_df["Probability"].apply(lambda x: f"{x * 100:.2f}%")
            ),
            use_container_width=True
        )

    else:
        chart_df = pd.DataFrame({
            "Class": ["Confidence"],
            "Probability": [confidence]
        })

        st.bar_chart(chart_df, x="Class", y="Probability")

# ---------- Method Section ----------
st.markdown("---")
st.subheader("How the System Works")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("""
    ### 1. Upload
    The user uploads a brain MRI image.
    """)

with step2:
    st.markdown("""
    ### 2. Analyze
    The binary model checks for tumor presence.
    """)

with step3:
    st.markdown("""
    ### 3. Classify
    If tumor is detected, the second model predicts tumor type.
    """)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
Built with Streamlit, PyTorch, and TensorFlow/Keras | Medical Image Analysis Project
</div>
""", unsafe_allow_html=True)