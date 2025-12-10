"""
Helper functions for working with Google Gemini and emotion-aware prompts.
Adapted to this project (Streamlit + DeepFace).
"""

from collections import Counter
from typing import List, Optional, Tuple

import google.generativeai as genai
import streamlit as st


# Map emotion -> emoji (có thể điều chỉnh cho phù hợp)
emotion_emoji = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# Map emotion -> Vietnamese name
emotion_vietnamese = {
    "angry": "tức giận",
    "Anger": "tức giận",
    "disgust": "khó chịu",
    "Disgust": "khó chịu",
    "fear": "lo lắng",
    "Fear": "lo lắng",
    "happy": "vui vẻ",
    "Happy": "vui vẻ",
    "sad": "buồn bã",
    "Sad": "buồn bã",
    "surprise": "ngạc nhiên",
    "Surprise": "ngạc nhiên",
    "neutral": "bình thường",
    "Neutral": "bình thường",
    "Contempt": "không hài lòng",
}


def create_emotion_intro(emotion: str) -> str:
    """
    Tạo câu giới thiệu cảm xúc khách hàng bằng tiếng Việt.
    Dùng cho bối cảnh nhà hàng - thông báo cho nhân viên.
    
    Args:
        emotion: Emotion name (e.g., "Happy", "Sad")
    
    Returns:
        Vietnamese introduction sentence for restaurant staff
    """
    emotion_vi = emotion_vietnamese.get(emotion, emotion)
    return f"Khách hàng đang có biểu hiện {emotion_vi}. "

def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from various sources."""

    import os
    from pathlib import Path

    # Try loading from .env file first
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        # python-dotenv chưa được cài, bỏ qua
        pass
    except Exception:
        # Có lỗi khi load .env, bỏ qua
        pass

    # Try secrets first (Streamlit secrets)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass

    # Try environment variable (sau khi đã load .env)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    # Try session state (user input)
    if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
        return st.session_state.gemini_api_key

    return None


def init_gemini(api_key: str, model_name: Optional[str] = None):
    """Initialize Gemini API with API key and choose a sensible default model."""

    if not api_key:
        return None, "Chưa có GEMINI_API_KEY"

    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()

        available_models = []
        free_tier_models = []

        for model in models:
            if "generateContent" in model.supported_generation_methods:
                model_name_clean = model.name.replace("models/", "")

                # Lọc bớt experimental
                if "-exp" not in model_name_clean and "experimental" not in model_name_clean.lower():
                    available_models.append(model_name_clean)

                    if any(
                        ft_model in model_name_clean
                        for ft_model in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                    ):
                        free_tier_models.append(model_name_clean)

        preferred_models = [
            "gemini-1.5-flash",  # nhanh, free tier tốt
            "gemini-1.5-pro",
            "gemini-pro",
        ]

        selected_model = None

        if model_name and model_name in available_models:
            selected_model = model_name
        else:
            for pref_model in preferred_models:
                if pref_model in free_tier_models:
                    selected_model = pref_model
                    break

            if not selected_model and free_tier_models:
                selected_model = free_tier_models[0]

            if not selected_model and available_models:
                selected_model = available_models[0]

        if not selected_model:
            return None, "Không tìm thấy model Gemini nào khả dụng"

        model = genai.GenerativeModel(selected_model)
        return model, selected_model
    except Exception as e:
        return None, f"Lỗi khi khởi tạo Gemini API: {e}"