import numpy as np
from deepface import DeepFace
from PIL import Image
import streamlit as st


def analyze_emotion(image: Image.Image):
    """
    Phân tích cảm xúc từ ảnh (PIL Image) bằng DeepFace.
    Trả về kết quả phân tích hoặc None nếu có lỗi.
    """
    try:
        # Chuyển PIL Image -> numpy array (RGB)
        img_rgb = np.array(image)

        # DeepFace chấp nhận numpy array (BGR hoặc RGB tùy backend),
        # ở đây ta truyền trực tiếp array RGB.
        result = DeepFace.analyze(
            img_path=img_rgb,
            actions=["emotion"],
            enforce_detection=False,
        )

        # DeepFace có thể trả về list hoặc dict tùy version
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        return result
    except Exception as e:
        st.error(f"Lỗi khi phân tích cảm xúc: {e}")
        return None


