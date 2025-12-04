#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra Gemini API có hoạt động không.
"""

import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Đã load .env từ: {env_path}")
    else:
        print(f"⚠️ Không tìm thấy file .env tại: {env_path}")
except ImportError:
    print("⚠️ python-dotenv chưa được cài. Chạy: pip install python-dotenv")

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Không tìm thấy GEMINI_API_KEY trong environment variables!")
    print("Vui lòng:")
    print("1. Tạo file .env với: GEMINI_API_KEY=your_key")
    print("2. Hoặc export GEMINI_API_KEY=your_key")
    sys.exit(1)

print(f"✅ Tìm thấy API key: {api_key[:10]}...{api_key[-5:]}")

# Test Gemini API
try:
    import google.generativeai as genai
    
    print("\n🔄 Đang khởi tạo Gemini...")
    genai.configure(api_key=api_key)
    
    # List models
    print("📋 Đang lấy danh sách models...")
    models = genai.list_models()
    available = [m.name.replace("models/", "") for m in models
                if "generateContent" in m.supported_generation_methods]

    print(f"✅ Tìm thấy {len(available)} models khả dụng")

    preferred = [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    selected = next((m for m in preferred if m in available), None)

    if not selected:
        selected = available[0]

    print(f"✅ Chọn model: {selected}")
    model = genai.GenerativeModel(selected)
    prompt = """Bạn là một trợ lý cảm xúc chuyên nghiệp. Hiện tại người dùng đang có cảm xúc: happy 😄

Hãy đưa ra một gợi ý hỗ trợ ngắn gọn và phù hợp với cảm xúc hiện tại của người dùng. Format (trả lời bằng tiếng Việt):

**Tiêu đề:** [Tiêu đề ngắn gọn về cảm xúc happy]

**Gợi ý:** [2-3 câu gợi ý ngắn gọn, thực tế, phù hợp với cảm xúc này]

Hãy trả lời ngay:"""
    
    response = model.generate_content(prompt)
    
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        text = response.candidates[0].content.parts[0].text
    else:
        text = str(response)
    
    print("\n" + "="*60)
    print("✅ RESPONSE TỪ GEMINI:")
    print("="*60)
    print(text)
    print("="*60)
    print(f"\n✅ Test thành công! Response có {len(text)} ký tự.")
    
except Exception as e:
    print(f"\n❌ LỖI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

