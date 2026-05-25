"""Streamlit dashboard for SmartBite food quality demo."""

import streamlit as st
import numpy as np
from PIL import Image
from smartbite.inference.engine import InferenceEngine, InferenceConfig
from smartbite.models.classifier import FoodClassifier


@st.cache_resource
def load_engine():
    model = FoodClassifier(num_classes=210, pretrained=False)
    config = InferenceConfig(device="cpu")
    return InferenceEngine(model, config)


def main():
    st.set_page_config(page_title="SmartBite Food Quality", layout="wide")
    st.title("SmartBite Food Quality Inspector")
    st.markdown("Upload a food image for AI-powered freshness and quality analysis")

    uploaded = st.file_uploader("Choose food image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        img_array = np.array(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        engine = load_engine()
        result = engine.predict(img_array)

        col1, col2, col3 = st.columns(3)
        col1.metric("Freshness Score", f"{result.freshness.overall:.2f}")
        col2.metric("Quality Grade", result.freshness.quality_grade.value)
        col3.metric("Confidence", f"{result.freshness.confidence:.2%}")

        st.subheader("Freshness Breakdown")
        st.progress(result.freshness.overall)
        st.text(f"Appearance: {result.freshness.appearance:.2f}")
        st.text(f"Texture: {result.freshness.texture:.2f}")
        st.text(f"Color: {result.freshness.color:.2f}")
        st.text(f"Spoilage: {result.freshness.spoilage_level.name}")
        st.text(f"Inference: {result.inference_time_ms:.1f} ms")


if __name__ == "__main__":
    main()