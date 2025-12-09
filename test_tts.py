import edge_tts
import asyncio

async def test():
    try:
        voices = await edge_tts.list_voices()
        print(len(voices))
    except Exception as e:
        print("VOICE ERROR:", e)

asyncio.run(test())
