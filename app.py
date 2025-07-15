import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
import pandas as pd
from openai import OpenAI
import openai
import os

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Depression-Aware Chatbot", layout="centered")

# === OPENAI CLIENT ===
#client = OpenAI(api_key="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")  # replace with your actual key

client = OpenAI(
    api_key="3bea1b3d2487149914651210dccff508b06e18b53831097e209b983bce20d731",
    base_url="https://api.together.xyz/v1"
)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD ROBERTA MODEL AND TOKENIZER ===
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta_depression_tokenizer")
    model = RobertaForSequenceClassification.from_pretrained("roberta_depression_model")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# === DEPRESSION DETECTION ===
def detect_depression(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        label = torch.argmax(logits, dim=1).item()
    return label  # 0 = not depressed, 1 = depressed

# === GPT REPLY ===
def get_llama_reply(user_message):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # or another model hosted by Together.ai
            messages=[
                {"role": "system", "content": "You are a kind, helpful, and empathetic chatbot."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ LLaMA API Error: {str(e)}")
        return "Sorry, I'm having trouble thinking right now. Try again later."

# === SESSION STATE ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === UI LAYOUT ===
st.title("🧠 Smart Depression-Aware Chatbot")
user_input = st.text_input("You:", key="input")

# === PROCESS INPUT ===
if st.button("Send") and user_input.strip():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Detect depression
    label = detect_depression(user_input)
    flag = "❌ Depressed" if label == 1 else "✅ Not Depressed"
    
    if label == 1:
        st.warning("⚠️ This message may indicate signs of depression. Please check on your chat partner.")
    
    # Get reply from ChatGPT
    bot_reply = get_llama_reply(user_input)

    # Append to history
    st.session_state.chat_history.append({
        "time": timestamp,
        "sender": "You",
        "text": user_input,
        "label": flag
    })
    st.session_state.chat_history.append({
        "time": timestamp,
        "sender": "Bot",
        "text": bot_reply,
        "label": ""
    })

# === CHAT DISPLAY ===
st.markdown("### 💬 Chat History")
for msg in st.session_state.chat_history:
    bubble = f"""
    <div style='
        background-color:{"#DCF8C6" if msg["sender"]=="You" else "#F1F0F0"};
        padding:10px;
        border-radius:10px;
        margin:5px;
        width:fit-content;
        max-width:80%;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);'
    >
        <b>{msg["sender"]}:</b> {msg["text"]}<br>
        <sub>{msg["label"]}</sub>
    </div>
    """
    st.markdown(bubble, unsafe_allow_html=True)

# === SAVE CHAT TO CSV ===
log_df = pd.DataFrame(st.session_state.chat_history)
log_df.to_csv("chat_logs.csv", index=False)
