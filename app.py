import streamlit as st

from components.camera_auto import render_camera_auto
from services.gemini_service import get_gemini_api_key


st.set_page_config(
    page_title="Real-time Emotion Detection",
    page_icon="😄",
    layout="wide",
)


def render_navbar_and_hero():
    """Navbar + hero section styled giống mẫu landing."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        .custom-navbar {
            top: 0;
            z-index: 999;
            background: #ffffff;
            border-bottom: 1px solid #e5e7eb;
            padding: 12px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .nav-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-icon {
            width: 46px;
            height: 46px;
            border-radius: 50%;
            background: #2563eb;
            color: white;
            display: grid;
            place-items: center;
            font-weight: 700;
            font-size: 18px;
        }
        .brand-text {
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .nav-links a {
            color: #475569;
            text-decoration: none;
            font-weight: 600;
            font-size: 18px;
        }
        .nav-links a:hover { 
            color: #2563eb; 
        }
        .nav-login {
            padding: 8px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            text-decoration: none;
            color: #0f172a;
            font-weight: 600;
            transition: all 0.2s ease;
            font-size: 20px;
            text-decoration: none;
        }
        .nav-login:hover {
            border-color: #2563eb;
            color: #2563eb;
        }
        .hero-wrapper {
            padding: 28px;
            border-radius: 20px;
            background: linear-gradient(90deg, #e5f2ff 0%, #e6ffee 100%);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            align-items: center;
            border: 1px solid #dbeafe;
        }
        .hero-text h1 {
            margin: 0 0 12px;
            font-size: 32px;
            color: #0f172a;
        }
        .hero-text p {
            margin: 0 0 12px;
            color: #475569;
            font-size: 18px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(37, 99, 235, 0.1);
            color: #2563eb;
            border-radius: 999px;
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 12px;
        }
        .hero-actions {
            display: flex;
            gap: 12px;
            margin-top: 12px;
        }
        .cta-btn {
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid #2563eb;
            text-decoration: none;
            font-weight: 700;
            font-size: 18px;
        }
        .cta-primary {
            background: #2677d9;
            color: #ffffff;
        }
        .cta-secondary {
            background: #ffffff;
            color: #2563eb;
        }
        .hero-illustration {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .hero-emoji {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: #ffffff;
            display: grid;
            place-items: center;
            font-size: 48px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        }
        .hero-emoji-row {
            display: flex;
            gap: 10px;
            font-size: 22px;
        }
        </style>
        <div class="custom-navbar">
            <div class="nav-left">
                <div class="brand-icon">AI</div>
                <div class="brand-text">AI Emotion</div>
            </div>
            <a class="nav-login" href="#dang-nhap">Đăng nhập</a>
        </div>

        <div class="hero-wrapper" id="tong-quan">
            <div class="hero-text">
                <div class="hero-badge">Real-time AI Demo</div>
                <h1>Phát hiện Emotion thời gian thực trên Web</h1>
                <p>
                    Ứng dụng demo sử dụng DeepFace để detect cảm xúc từ camera hoặc ảnh upload.
                    Do giới hạn của Streamlit, camera hoạt động theo kiểu chụp từng frame (gần real-time),
                    không phải video stream liên tục như ứng dụng desktop.
                </p>
                <div class="hero-actions">
                    <a class="cta-btn cta-primary" href="#camera">Bắt đầu ngay</a>
                    <a class="cta-btn cta-secondary" href="#ve-thoi">Tìm hiểu thêm</a>
                </div>
            </div>
            <div class="hero-illustration">
                <div class="hero-emoji">🙂</div>
                <div class="hero-emoji-row">😐 😊 😡 😢 😲</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Áp dụng max-width chung trước khi render layout tùy chỉnh
    render_navbar_and_hero()

    api_key = get_gemini_api_key()

    st.markdown('<div id="camera"></div>', unsafe_allow_html=True)
    st.markdown("---")

    render_camera_auto(interval_seconds=15)


if __name__ == "__main__":
    main()

