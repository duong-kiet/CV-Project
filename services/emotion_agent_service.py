import sqlite3
import os

# ==========================
# SQLite MEMORY – Lưu lịch sử cảm xúc khách hàng
# ==========================

DB_PATH = "emotion_memory.db"

def init_memory_db():
    """Khởi tạo database nếu chưa có."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            emotion TEXT,
            intensity REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_emotion_history(user_id: str, emotion: str, intensity: float):
    """Lưu cảm xúc vào SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO emotion_history (user_id, emotion, intensity) VALUES (?, ?, ?)",
        (user_id, emotion, intensity),
    )
    conn.commit()
    conn.close()

def get_recent_emotions(user_id: str, limit: int = 5):
    """Lấy lịch sử cảm xúc gần nhất của khách."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT emotion, intensity, timestamp FROM emotion_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# ==========================
# HÀM CHÍNH: CHATBOT HỖ TRỢ NHÀ HÀNG
# ==========================

def generate_advice_with_memory_from_result(
    model,
    dominant_emotion: str,
    emotions: dict,
    user_id: str = "default_user"
):
    """
    PHIÊN BẢN CHATBOT NHÀ HÀNG — TIẾNG VIỆT

    Nhiệm vụ:
    - Giải thích ngắn gọn trạng thái cảm xúc khách hàng
    - Đưa ra hướng dẫn rõ ràng cho nhân viên phục vụ
    - Không nói như chuyên gia tâm lý
    - Không trấn an khách
    - Tập trung vào chất lượng phục vụ và cải thiện trải nghiệm nhà hàng
    """

    # 1. Lưu vào SQLite
    emotion_intensity = emotions.get(dominant_emotion, 1.0)
    save_emotion_history(user_id, dominant_emotion, float(emotion_intensity))

    # 2. Lấy lịch sử cảm xúc gần nhất
    recent_history = get_recent_emotions(user_id)
    history_text = "\n".join(
        [f"- {e} (mức {round(i,2)}) lúc {t}" for e, i, t in recent_history]
    ) or "Chưa có lịch sử cảm xúc nào."

    # 3. Prompt tiếng Việt
    prompt = f"""
Bạn là **RestaurantAI**, trợ lý nội bộ cho nhà hàng.

Hệ thống vừa phát hiện cảm xúc của khách:
- Cảm xúc chính: **{dominant_emotion}**
- Điểm các cảm xúc: {emotions}

Lịch sử cảm xúc gần đây của khách:
{history_text}

Nhiệm vụ của bạn:
1. Mô tả ngắn gọn trạng thái cảm xúc hiện tại của khách.
2. Đưa ra hướng dẫn cụ thể, rõ ràng cho nhân viên phục vụ phải làm NGAY BÂY GIỜ.
3. Đề xuất bước xử lý tiếp theo nếu cần (ví dụ: kiểm tra món ăn, đổi bàn, xin lỗi, hỏi thăm nhẹ nhàng…)
4. Không đưa lời khuyên mang tính tâm lý cá nhân.
5. Không an ủi khách.
6. Giọng điệu như trợ lý nội bộ nhà hàng — ngắn gọn, tập trung vào nghiệp vụ.

Hãy trả lời bằng tiếng Việt.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi tạo phản hồi từ AI: {e}"


# Khởi tạo database lần đầu
if not os.path.exists(DB_PATH):
    init_memory_db()
