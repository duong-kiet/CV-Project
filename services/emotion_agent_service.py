import sqlite3
import os

# ==========================
# SQLite MEMORY – Lưu lịch sử cảm xúc khách hàng
# ==========================

DB_PATH = "database/emotion_memory.db"

def init_memory_db():
    """Khởi tạo database nếu chưa có."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            phase INTEGER,
            emotion TEXT,
            intensity REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_emotion_history(user_id: str, phase: int, emotion: str, intensity: float):
    """Lưu cảm xúc vào SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO emotion_history (user_id, phase, emotion, intensity) VALUES (?, ?, ?, ?)",
        (user_id, phase, emotion, intensity),
    )
    conn.commit()
    conn.close()

def get_recent_emotions(user_id: str, limit: int = 5):
    """Lấy lịch sử cảm xúc gần nhất của khách."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT phase, emotion, intensity, timestamp FROM emotion_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# ==========================
# PHASE MỚI – thêm 1 phase số 0
# ==========================

PHASE_LABELS = {
    0: "khi khách vừa bước vào nhà hàng (ấn tượng ban đầu)",
    1: "khi khách thấy thái độ phục vụ của nhân viên bồi bàn",
    2: "khi khách nhìn thấy món ăn (đánh giá trình bày)",
    3: "khi khách đang ăn món ăn",
    4: "khi khách trò chuyện (không liên quan dịch vụ)",
    5: "khi khách thanh toán"
}


# ==========================
# HÀM CHÍNH: CHATBOT HỖ TRỢ NHÀ HÀNG
# ==========================

def generate_advice_with_memory_from_result(
    model,
    dominant_emotion: str,
    emotions: dict,
    phase: int,                 # <<< có phase 0 mới
    user_id: str = "default_user"
):
    """
    Bản AI nhà hàng – có 6 phase, bổ sung phase 0 mới.

    Phase:
        0 – khách vừa vào nhà hàng (ấn tượng ban đầu)
        1 – khách đánh giá thái độ phục vụ
        2 – khách nhìn món ăn
        3 – khách ăn món ăn
        4 – khách trò chuyện
        5 – khách thanh toán
    """

    # 1. Lưu vào SQLite
    emotion_intensity = emotions.get(dominant_emotion, 1.0)
    save_emotion_history(user_id, phase, dominant_emotion, float(emotion_intensity))

    # 2. Lịch sử cảm xúc gần nhất
    recent_history = get_recent_emotions(user_id)
    history_text = "\n".join(
        [f"- Phase {p}: {e} (mức {round(i,2)}) lúc {t}" for p, e, i, t in recent_history]
    ) or "Chưa có lịch sử cảm xúc nào."

    # 3. Mô tả Phase
    phase_desc = PHASE_LABELS.get(phase, "trạng thái bình thường")

    # 4. Prompt xử lý theo từng phase
    prompt = f"""
Bạn là RestaurantAI – trợ lý nội bộ dành cho nhân viên nhà hàng.

Hệ thống vừa ghi nhận cảm xúc của khách {phase_desc}:
- Cảm xúc chính: {dominant_emotion}
- Điểm các cảm xúc: {emotions}
- Phase hiện tại: {phase}

Lịch sử cảm xúc gần đây:
{history_text}

Nhiệm vụ:
1. Nhận xét ngắn gọn cảm xúc của khách trong đúng giai đoạn này.
2. Đưa ra hành động CỤ THỂ nhân viên cần làm NGAY.
3. Tập trung theo từng phase:
    - Phase 0 → đánh giá ấn tượng ban đầu khi khách bước vào.
    - Phase 1 → đánh giá thái độ nhân viên bồi bàn.
    - Phase 2 → đánh giá trình bày món ăn.
    - Phase 3 → đánh giá chất lượng món ăn khi khách đang ăn.
    - Phase 4 → chỉ quan sát, không suy diễn thành đánh giá dịch vụ.
    - Phase 5 → đánh giá dịch vụ thanh toán.
4. Không an ủi khách.
5. Không đưa lời khuyên mang tính tâm lý cá nhân.
6. Giọng điệu sắc nét – nội bộ – tập trung vào hành động.
7. Trả lời tối đa 5 câu.
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
