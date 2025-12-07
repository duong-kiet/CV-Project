import streamlit as st
from PIL import Image

from services.deepface_service import analyze_emotion


def render_upload_image():
    """
    Chế độ upload ảnh từ máy để phân tích cảm xúc.
    """
    st.subheader("Upload ảnh từ máy")
    uploaded_file = st.file_uploader(
        "Chọn một ảnh (jpg, jpeg, png)...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã upload", use_container_width=True)

        if st.button("Phân tích cảm xúc"):
            with st.spinner("Đang phân tích cảm xúc..."):
                result = analyze_emotion(image)

            if result is not None:
                dominant_emotion = result.get("dominant_emotion")
                emotions = result.get("emotion", {})

                st.success(f"**Cảm xúc chính**: {dominant_emotion}")

                if emotions:
                    st.subheader("Chi tiết các cảm xúc")
                    st.bar_chart(emotions)


