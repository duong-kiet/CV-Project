import streamlit as st
from PIL import Image

from services.deepface_service import analyze_emotion


def render_camera_manual():
    """
    Chế độ chụp ảnh từ camera bằng tay (st.camera_input).
    """
    st.subheader("Chụp ảnh từ camera")
    st.write("Nhấn **Take photo** để chụp và phân tích cảm xúc.")

    camera_image = st.camera_input("Camera")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Ảnh từ camera", use_column_width=True)

        with st.spinner("Đang phân tích cảm xúc..."):
            result = analyze_emotion(image)

        if result is not None:
            dominant_emotion = result.get("dominant_emotion")
            emotions = result.get("emotion", {})

            st.success(f"**Cảm xúc chính**: {dominant_emotion}")

            if emotions:
                st.subheader("Chi tiết các cảm xúc")
                st.bar_chart(emotions)


