"""
Text-to-Speech service using edge-tts (Microsoft Edge TTS).
High quality Vietnamese voices, free, no API key needed.
"""

import asyncio
import tempfile
import os
import streamlit as st
import edge_tts


# Cache for Vietnamese voices
_vietnamese_voices = None


async def get_vietnamese_voices():
    """Get list of Vietnamese voices."""
    global _vietnamese_voices
    if _vietnamese_voices is None:
        try:
            voices = await edge_tts.list_voices()
            # Filter Vietnamese voices
            _vietnamese_voices = [
                voice for voice in voices 
                if voice["Locale"].startswith("vi")
            ]
            # Prefer female voices (usually sound more natural)
            _vietnamese_voices.sort(key=lambda x: "Female" in x.get("Gender", ""), reverse=True)
        except Exception as e:
            st.warning(f"Không thể lấy danh sách giọng: {e}")
            _vietnamese_voices = []
    return _vietnamese_voices


async def get_best_vietnamese_voice():
    """Get the best Vietnamese voice available."""
    voices = await get_vietnamese_voices()
    if voices:
        # Prefer female voices, then by name
        return voices[0]["Name"]
    # Fallback to any Vietnamese voice
    return "vi-VN-HoaiMyNeural"  # Good quality Vietnamese female voice


def text_to_speech(text: str, lang: str = "vi", slow: bool = False) -> bool:
    """
    Convert text to speech and play using system player (blocking call).
    For Streamlit, use text_to_speech_file() instead.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: "vi" for Vietnamese)
        slow: Whether to speak slowly (default: False)
    
    Returns:
        True if successful, False otherwise
    """
    audio_file = text_to_speech_file(text, lang, slow)
    if not audio_file:
        return False
    
    try:
        # Play audio using system player
        import platform
        system = platform.system()
        
        if system == "Linux":
            os.system(f"mpg123 -q {audio_file} 2>/dev/null || aplay {audio_file} 2>/dev/null || paplay {audio_file} 2>/dev/null")
        elif system == "Darwin":  # macOS
            os.system(f"afplay {audio_file}")
        elif system == "Windows":
            os.system(f'powershell -c (New-Object Media.SoundPlayer "{audio_file}").PlaySync()')
        else:
            # Fallback: try common players
            os.system(f"mpg123 -q {audio_file} 2>/dev/null || play {audio_file} 2>/dev/null")
        
        # Cleanup
        cleanup_audio_file(audio_file)
        return True
    except Exception as e:
        st.error(f"Lỗi khi phát audio: {e}")
        cleanup_audio_file(audio_file)
        return False


async def _text_to_speech_async(text: str, slow: bool = False) -> str:
    """Internal async function to generate TTS audio file."""
    try:
        # Get best Vietnamese voice
        voice = await get_best_vietnamese_voice()
        
        # Adjust rate if slow
        rate = "-20%" if slow else "+0%"
        
        # Generate TTS
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
            await communicate.save(tmp_path)
        
        return tmp_path
    except Exception as e:
        st.error(f"Lỗi khi tạo audio: {e}")
        return None


def text_to_speech_file(text: str, lang: str = "vi", slow: bool = False) -> str:
    """
    Convert text to speech and return audio file path.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: "vi" for Vietnamese)
        slow: Whether to speak slowly (default: False)
    
    Returns:
        Path to audio file, or None if error
    """
    if not text or not text.strip():
        return None
    
    try:
        # Clean text - remove markdown formatting
        clean_text = text
        # Remove markdown headers
        clean_text = clean_text.replace("###", "").replace("##", "").replace("#", "")
        # Remove markdown bold
        clean_text = clean_text.replace("**", "")
        # Remove markdown italic
        clean_text = clean_text.replace("*", "")
        # Remove extra whitespace
        clean_text = " ".join(clean_text.split())
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_text_to_speech_async(clean_text, slow))
        finally:
            loop.close()
    except Exception as e:
        st.error(f"Lỗi khi tạo audio file: {e}")
        return None


def text_to_speech_async(text: str, lang: str = "vi", slow: bool = False):
    """
    Convert text to speech asynchronously (non-blocking).
    Note: This returns a coroutine, not a thread.
    """
    return _text_to_speech_async(text, slow)


def estimate_speech_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate speech duration based on text length.
    
    Args:
        text: Text to estimate
        words_per_minute: Average speaking speed (default: 150 WPM)
    
    Returns:
        Estimated duration in seconds
    """
    if not text:
        return 0
    
    # Count words (approximate)
    word_count = len(text.split())
    # Calculate duration
    duration_minutes = word_count / words_per_minute
    duration_seconds = duration_minutes * 60
    
    # Add buffer for pauses
    return duration_seconds + 1  # Add 1 second buffer


def cleanup_audio_file(file_path):
    """Clean up function (not needed for pyttsx3, kept for compatibility)."""
    pass

