import streamlit as st
import requests
import io

st.set_page_config(page_title="FinBreaker", layout="centered")
st.title("FinBreaker ☕💹")

st.markdown("""
Morning news breaker for finance!
This assistant answers: **"What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"**

- Upload a voice question or type your query below.
- Get a spoken and text market brief.
""")

# Voice input
st.header("🎤 Voice Input")
audio_file = st.file_uploader("Upload a WAV audio file with your question", type=["wav"])

if audio_file and st.button("Transcribe & Analyze (Voice)"):
    st.info("Transcribing...")
    audio_bytes = audio_file.read()
    # Call orchestrator for full workflow
    resp = requests.post(
        "http://localhost:8000/orchestrator/morning_brief",
        json={"audio": audio_bytes.decode('ISO-8859-1')}
    )
    result = resp.json()
    transcript = result.get("transcript", "")
    st.write(f"**Transcript:** {transcript}")
    answer = result.get("answer", "No answer.")
    st.success(answer)
    audio_out = result.get("audio", None)
    if audio_out:
        st.audio(io.BytesIO(bytes(audio_out, encoding='ISO-8859-1')), format="audio/wav")

# Text input
st.header("⌨️ Text Input")
user_query = st.text_area("Type your market question", "What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?")
if st.button("Analyze (Text)"):
    st.info("Generating market brief...")
    resp = requests.post(
        "http://localhost:8000/orchestrator/morning_brief",
        json={"question": user_query}
    )
    result = resp.json()
    answer = result.get("answer", "No answer.")
    st.success(answer)
    audio_out = result.get("audio", None)
    if audio_out:
        st.audio(io.BytesIO(bytes(audio_out, encoding='ISO-8859-1')), format="audio/wav")

st.markdown("---")
st.caption("Open-source multi-agent finance assistant. Powered by FastAPI, LangChain, Deepgram, pyttsx3, and Streamlit.")
