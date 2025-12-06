from typing import Optional, Dict, Any

from PIL import Image

from services.deepface_service import analyze_emotion
from services.gemini_service import (
    get_gemini_api_key,
    init_gemini,
)
from services.vector_db_service import (
    upsert_emotion_memory,
    search_emotion_memory,
)

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # fallback khi không chạy trong Streamlit


def _generate_advice_with_memory_core(
    model,
    user_id: str,
    dominant_emotion: str,
    emotions: Dict[str, Any],
) -> Optional[str]:
    """
    Logic chung:
    - Lấy lịch sử từ SQLite TRƯỚC (để lấy cảm xúc cũ thật sự)
    - Lưu lần đo này vào SQLite SAU (để lần sau có thể query)
    - Dùng Gemini sinh lời khuyên có context
    """
    # 1. Lấy lịch sử cảm xúc gần đây TRƯỚC (trước khi insert cảm xúc hiện tại)
    try:
        similar_memories = search_emotion_memory(
            user_id=user_id,
            current_emotion=dominant_emotion,
            top_k=1,
        )

        previous_snapshot = similar_memories[0] if similar_memories else None
        previous_payload = previous_snapshot.get("payload") if isinstance(previous_snapshot, dict) else {}

        prev_text = previous_payload.get("text") if isinstance(previous_payload, dict) else None
        prev_dominant = previous_payload.get("dominant_emotion") if isinstance(previous_payload, dict) else None
        prev_timestamp = previous_payload.get("timestamp") if isinstance(previous_payload, dict) else None

        if prev_text or prev_dominant:
            context_lines = []
            if prev_timestamp:
                context_lines.append(f"- Thời điểm gần nhất trước đó: {prev_timestamp}.")
            if prev_dominant:
                context_lines.append(
                    f"- Khi đó, cảm xúc chính của người dùng là: {prev_dominant}."
                )
            if prev_text:
                context_lines.append(f"- Mô tả chi tiết lần trước: {prev_text}")

            context_block = "\n".join(context_lines)
        else:
            context_block = ""
    except Exception as e:
        context_block = ""
        print("ERROR search_emotion_memory:", e)

    # 2. Lưu lần đo này vào SQLite SAU (sau khi đã query lấy cảm xúc cũ)
    try:
        upsert_emotion_memory(
            user_id=user_id,
            dominant_emotion=dominant_emotion,
            emotions=emotions,
        )
    except Exception as e:
        # Log rõ lỗi để biết vì sao emotion_memory trống
        print("ERROR upsert_emotion_memory:", e)

    # 3. Xây dựng prompt cho một lần gọi model duy nhất
    if context_block.strip():
        # Có lịch sử trong SQLite: dùng cả cảm xúc cũ + hiện tại
        prompt = f"""
Bạn là trợ lý cảm xúc có trí nhớ dài hạn.

Dưới đây là một số lần đo cảm xúc gần đây của người dùng (từ ảnh khuôn mặt):
{context_block}

Hiện tại, cảm xúc chính của người dùng là: {dominant_emotion}.

Hãy trả lời một phiên bản ngắn gọn, tập trung vào:
- Nhận xét xu hướng cảm xúc gần đây nếu có (1–2 câu).
- Gợi ý hành động cụ thể, thực tế, phù hợp với cảm xúc hiện tại (2–3 câu).

Trả lời bằng tiếng Việt.
"""
    else:
        # Không có lịch sử: chỉ dùng cảm xúc hiện tại
        prompt = f"""
Bạn là một trợ lý cảm xúc chuyên nghiệp.

Hiện tại người dùng đang có cảm xúc chính là: {dominant_emotion}.

Hãy:
- Mô tả ngắn gọn trạng thái cảm xúc hiện tại (1–2 câu).
- Đưa ra 2–3 câu ngắn gọn gợi ý thực tế, nhẹ nhàng, giúp người dùng đối diện và điều chỉnh cảm xúc này.

Trả lời bằng tiếng Việt.
"""

    print("Prompt generate advice with memory core:", prompt)

    try:
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        # Trả về message lỗi rõ ràng để hiển thị lên UI
        return f"⚠️ Lỗi khi gọi Gemini: {str(e)[:200]}"


def analyze_image_and_generate_advice(
    image: Image.Image,
    user_id: str = "default_user",
) -> Optional[str]:
    """
    Pipeline tiện dụng:
    - Nhận ảnh (PIL Image)
    - DeepFace phân tích cảm xúc
    - Lưu & search trong SQLite
    - Gọi Gemini để sinh lời khuyên có context
    """
    # 1. Phân tích cảm xúc từ ảnh
    result = analyze_emotion(image)
    if not result:
        return None

    dominant_emotion = result.get("dominant_emotion")
    emotions = result.get("emotion", {})

    if not dominant_emotion:
        return None

    # 2. Khởi tạo Gemini model
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ Chưa tìm thấy GEMINI_API_KEY, không thể gọi Gemini."

    model, model_info = init_gemini(api_key)
    if not model:
        return f"⚠️ Lỗi khởi tạo Gemini model: {model_info}"

    return _generate_advice_with_memory_core(
        model=model,
        user_id=user_id,
        dominant_emotion=dominant_emotion,
        emotions=emotions,
    )


def generate_advice_with_memory_from_result(
    model,
    dominant_emotion: str,
    emotions: Dict[str, Any],
    user_id: str = "default_user",
) -> Optional[str]:
    """
    Dùng khi bạn đã có:
    - model Gemini (đã init sẵn)
    - dominant_emotion + emotions (từ DeepFace)

    → Gọi SQLite + Gemini để sinh lời khuyên có context.
    """
    if not dominant_emotion:
        return None

    return _generate_advice_with_memory_core(
        model=model,
        user_id=user_id,
        dominant_emotion=dominant_emotion,
        emotions=emotions,
    )

