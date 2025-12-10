from typing import Optional, Dict, Any, List
import numpy as np

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
    similarity: Optional[float] = None,
    face_embedding: Optional[np.ndarray] = None,
    box: Optional[List[int]] = None,
) -> Optional[str]:
    """
    Logic chung:
    - Lấy lịch sử từ SQLite TRƯỚC (để lấy cảm xúc cũ thật sự)
    - Lưu lần đo này vào SQLite SAU (để lần sau có thể query)
    - Dùng Gemini sinh lời khuyên có context
    
    Args:
        model: Gemini model đã khởi tạo
        user_id: ID người dùng (face_id)
        dominant_emotion: Cảm xúc chính
        emotions: Dict các cảm xúc và xác suất
        similarity: Độ tương đồng với face đã biết
        face_embedding: Vector đặc trưng khuôn mặt (512-D)
        box: Bounding box [x1, y1, x2, y2]
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
            similarity=similarity,
            face_embedding=face_embedding,
            box=box,
        )
    except Exception as e:
        # Log rõ lỗi để biết vì sao emotion_memory trống
        print("ERROR upsert_emotion_memory:", e)

    # 3. Xây dựng prompt cho một lần gọi model duy nhất
    # Bối cảnh: AI agent hỗ trợ nhà hàng, đưa chỉ dẫn cho nhân viên phục vụ
    
    if context_block.strip():
        # Có lịch sử: khách quen hoặc đã theo dõi trước đó
        prompt = f"""
Bạn là trợ lý AI thông minh hỗ trợ nhân viên nhà hàng. Nhiệm vụ của bạn là phân tích cảm xúc khách hàng và đưa ra hướng dẫn cụ thể cho nhân viên phục vụ.

**Thông tin khách hàng (ID: {user_id}):**
Lịch sử cảm xúc gần đây:
{context_block}

**Cảm xúc hiện tại:** {dominant_emotion}

**Yêu cầu:** Hãy đưa ra hướng dẫn ngắn gọn cho nhân viên phục vụ:
1. **Nhận định tình trạng:** Phân tích ngắn gọn cảm xúc khách và xu hướng thay đổi (1-2 câu)
2. **Cách tiếp cận:** Gợi ý cách giao tiếp, thái độ phục vụ phù hợp (2-3 câu)
3. **Hành động cụ thể:** Đề xuất 1-2 hành động cụ thể (ví dụ: đề xuất món, tặng đồ uống, giảm âm lượng nhạc...)

Trả lời bằng tiếng Việt, ngắn gọn, thực tế và dễ áp dụng ngay.
"""
    else:
        # Khách mới, chưa có lịch sử
        prompt = f"""
Bạn là trợ lý AI thông minh hỗ trợ nhân viên nhà hàng. Nhiệm vụ của bạn là phân tích cảm xúc khách hàng và đưa ra hướng dẫn cụ thể cho nhân viên phục vụ.

**Khách hàng mới (ID: {user_id})**
**Cảm xúc hiện tại:** {dominant_emotion}

**Yêu cầu:** Hãy đưa ra hướng dẫn ngắn gọn cho nhân viên phục vụ:
1. **Nhận định:** Mô tả ngắn gọn trạng thái cảm xúc khách hàng (1 câu)
2. **Cách tiếp cận:** Gợi ý cách chào hỏi, thái độ phục vụ phù hợp với cảm xúc này (2 câu)
3. **Hành động cụ thể:** Đề xuất 1-2 hành động để nâng cao trải nghiệm khách hàng

**Lưu ý theo cảm xúc:**
- Happy/Neutral: Duy trì không khí tích cực, giới thiệu món đặc biệt
- Sad/Fear: Tạo không gian riêng tư, phục vụ chu đáo, nhẹ nhàng
- Anger/Disgust: Xử lý nhanh, lịch sự, tránh để khách chờ đợi
- Surprise: Tận dụng cơ hội tạo ấn tượng tốt

Trả lời bằng tiếng Việt, thực tế  và ngắn gọn nhất có thể.
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
    similarity: Optional[float] = None,
    face_embedding: Optional[np.ndarray] = None,
    box: Optional[List[int]] = None,
) -> Optional[str]:
    """
    Dùng khi bạn đã có:
    - model Gemini (đã init sẵn)
    - dominant_emotion + emotions (từ MLT model)
    - Các thông tin face: similarity, face_embedding, box

    → Gọi SQLite + Gemini để sinh lời khuyên có context.
    """
    if not dominant_emotion:
        return None

    return _generate_advice_with_memory_core(
        model=model,
        user_id=user_id,
        dominant_emotion=dominant_emotion,
        emotions=emotions,
        similarity=similarity,
        face_embedding=face_embedding,
        box=box,
    )
