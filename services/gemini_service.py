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
    "disgust": "ghê tởm",
    "fear": "sợ hãi",
    "happy": "vui vẻ",
    "sad": "buồn bã",
    "surprise": "ngạc nhiên",
    "neutral": "bình thường",
}


def create_emotion_intro(emotion: str) -> str:
    """
    Tạo câu giới thiệu cảm xúc bằng tiếng Việt.
    
    Args:
        emotion: Emotion name (e.g., "happy", "sad")
    
    Returns:
        Vietnamese introduction sentence
    """
    emotion_vi = emotion_vietnamese.get(emotion, emotion)
    return f"Khách hàng đang ở cảm xúc {emotion_vi}. "

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
                        for ft_model in ["gemini-1.5-flash", "gemini-1.5-pro"]
                    ):
                        free_tier_models.append(model_name_clean)

        preferred_models = [
            "gemini-1.5-flash",  # nhanh, free tier tốt
            "gemini-1.5-pro",
            
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


def analyze_emotion_pattern(emotion_list: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Analyze emotion pattern from a list of emotions."""

    if not emotion_list:
        return None, None

    emotion_counts = Counter(emotion_list)
    most_common_emotion, count = emotion_counts.most_common(1)[0]
    total_count = len(emotion_list)

    parts = []
    for emotion, cnt in emotion_counts.most_common():
        pct = (cnt / total_count) * 100
        parts.append(f"{emotion}: {cnt}/{total_count} ({pct:.1f}%)")

    pattern_description = ", ".join(parts)
    return most_common_emotion, pattern_description


def generate_suggestion_for_current_emotion(
    model, current_emotion: str, max_retries: int = 3
) -> Optional[str]:
    """Generate 1 đoạn gợi ý ngắn cho cảm xúc hiện tại với retry logic cho lỗi 429."""

    if not model or not current_emotion:
        return None

    emoji = emotion_emoji.get(current_emotion, "😐")

    prompt = f"""Bạn là một trợ lý cảm xúc chuyên nghiệp. Hiện tại người dùng đang có cảm xúc: {current_emotion} {emoji}

Hãy đưa ra một gợi ý hỗ trợ ngắn gọn và phù hợp với cảm xúc hiện tại của người dùng. Format (trả lời bằng tiếng Việt):

**Tiêu đề:** [Tiêu đề ngắn gọn về cảm xúc {current_emotion}]

**Gợi ý:** [2-3 câu gợi ý ngắn gọn, thực tế, phù hợp với cảm xúc này]

Hãy trả lời ngay:"""

    import time
    import re

    # Safety settings - cho phép tất cả content để tránh bị block
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
    for attempt in range(max_retries):
        try:
            # Gọi với safety settings để tránh bị block
            try:
                response = model.generate_content(
                    prompt,
                    safety_settings=safety_settings
                )
            except TypeError:
                # Một số model có thể không hỗ trợ safety_settings parameter
                response = model.generate_content(prompt)
            
            # Parse response - thử nhiều cách
            text = None
            
            # Cách 1: response.text (phổ biến nhất)
            if hasattr(response, "text"):
                text = response.text
            
            # Cách 2: response.candidates[0].content.parts[0].text
            if not text and hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content"):
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, "text"):
                            text = part.text
                    # Thử content.text
                    if not text and hasattr(candidate.content, "text"):
                        text = candidate.content.text
                # Thử candidate.text trực tiếp
                if not text and hasattr(candidate, "text"):
                    text = candidate.text
            
            # Cách 3: str(response) nếu có
            if not text:
                try:
                    text = str(response)
                    # Nếu là object representation, không dùng
                    if text.startswith("<") and "object" in text:
                        text = None
                except:
                    pass
            
            # Nếu có text, return
            if text and text.strip():
                return text.strip()
            
            # Nếu không có text, return error message với debug info
            debug_info = []
            if hasattr(response, "__dict__"):
                debug_info.append(f"response.__dict__ keys: {list(response.__dict__.keys())[:5]}")
            if hasattr(response, "candidates"):
                debug_info.append(f"candidates count: {len(response.candidates) if response.candidates else 0}")
            
            return f"""⚠️ **Response từ Gemini không có text!**

**Debug info:**
- Response type: `{type(response)}`
- Has 'text' attr: {hasattr(response, 'text')}
- Has 'candidates' attr: {hasattr(response, 'candidates')}
{chr(10).join(debug_info)}

**Có thể do:**
- Response bị block bởi safety settings
- Model không trả về content
- API response format thay đổi

**Thử:**
- Kiểm tra API key có đúng không
- Thử lại sau vài giây
- Kiểm tra quota/rate limits"""
        except Exception as e:
            error_msg = str(e)

            # Xử lý lỗi 429 (rate limit / quota exceeded)
            if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                # Extract retry delay nếu có trong error message
                retry_delay = 5  # Default delay
                if "retry in" in error_msg.lower():
                    delay_match = re.search(r"retry in ([\d.]+)s", error_msg.lower())
                    if delay_match:
                        retry_delay = float(delay_match.group(1)) + 1

                if attempt < max_retries - 1:
                    # Exponential backoff với max 60s
                    wait_time = min(retry_delay * (2 ** attempt), 60)
                    time.sleep(wait_time)
                    continue
                else:
                    return """⚠️ **Đã vượt quá hạn mức (quota) API miễn phí!**

**Nguyên nhân:**
- Bạn đã sử dụng hết quota miễn phí cho model hiện tại
- Model đang dùng có thể không có trong free tier

**Giải pháp:**
1. Đợi một chút (thường vài phút đến vài giờ) để quota reset
2. Kiểm tra usage tại: https://ai.dev/usage?tab=rate-limit
3. Xem rate limits tại: https://ai.google.dev/gemini-api/docs/rate-limits

**Lưu ý:** Model `gemini-1.5-flash` thường có quota tốt hơn cho free tier."""

            # Xử lý lỗi 404 (model not found)
            elif "404" in error_msg or "not found" in error_msg.lower():
                return "⚠️ Lỗi: Model không tìm thấy. Vui lòng kiểm tra API key và model name."

            # Các lỗi khác
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"⚠️ Xin lỗi, có lỗi xảy ra sau {max_retries} lần thử: {error_msg[:200]}"

    return "⚠️ Không thể kết nối với Gemini API. Vui lòng thử lại sau."


