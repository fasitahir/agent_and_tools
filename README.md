
# 💬 Smart Chat Assistant with Admin RAG Panel

This is a **Streamlit-powered chatbot application** built using **LangChain** and a **history-aware agent** with persistent conversation memory. It blends:

- 🧠 **Retrieval-Augmented Generation (RAG)** with BERT embeddings and ChromaDB
- 🧰 **Custom LangChain tools** (DNS lookup, weather, shell commands, Bookme assistant)
- 👨‍💻 **Admin interface** for uploading and embedding documents
- 🗨️ **ConversationBufferMemory** for contextual, multi-turn conversations

---

## 🚀 Features

| Feature               | Description |
|-----------------------|-------------|
| 🔍 **RAG Tool**        | Answers questions from your uploaded documents using BERT + Chroma |
| 👤 **History-Aware Agent** | Retains past interactions using `ConversationBufferMemory` |
| 📂 **Admin Panel**     | Upload `.pdf`/`.txt` files or manually embed lines using BERT |
| 🌐 **Tools**           | DNS lookup, weather, shell command execution, and Bookme.pk scraper |
| 📜 **Document QA**     | Combines embeddings + Gemini LLM for context-aware responses |

---

## 📦 Installation

1. **Clone the repo**:

```bash
git clone https://github.com/fasitahir/agent_and_tools.git
cd smart-chat-rag
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Add your environment variables**:

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key
```

(You can get your key from [Google AI Studio](https://makersuite.google.com/app))

---

## ▶️ Running the App

Launch Streamlit:

```bash
streamlit run app.py
```

Select your role from the sidebar:
- **Admin** – For document upload + embedding
- **User** – For chatting with tool-enabled, memory-aware agent

---

## 🧠 How It Works

- **LangChain Agent** uses `initialize_agent()` with:
  - Custom Tools (`@tool`)
  - Gemini 1.5 model (`ChatGoogleGenerativeAI`)
  - `ConversationBufferMemory` for remembering past messages
- **RAG Tool** uses BERT-based embeddings (via HuggingFace) + ChromaDB
- Admin uploads PDFs or texts → split → embedded → stored
- Users can chat using AI powered by tools + context + memory

---

## 📂 File Structure

```
├── app.py                   # Streamlit app (User + Admin UI)
├── bookme_scraper.py        # Scraper logic for Bookme.pk
├── db/
│   └── chroma_bert/         # Vector store
├── chat_history.json        # Persistent chat memory
├── requirements.txt         # Dependencies
├── .env                     # API key (not committed)
```

---

## 🧪 Sample Queries

Try these in the **User Chat Interface**:
- “What is the weather in Karachi?”
- “Create folder assignment_final”
- “Tell me what Bookme.pk says about bus ticket availability.”
- “Summarize the file I uploaded earlier.”

---

## 📌 Requirements

- Python 3.8+
- Streamlit
- LangChain
- HuggingFace Transformers
- Chroma
- torch
- requests
- python-dotenv

---

## 📜 License

MIT License – free to use, modify, and share.

---

## 💡 Credits

- Built using **LangChain**
- Uses **Gemini API** via Google AI Studio
- Developed with ❤️ for smart assistants and RAG systems
