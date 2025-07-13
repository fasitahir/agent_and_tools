import os
import socket
import subprocess
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import tool, initialize_agent
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import json
from dotenv import load_dotenv
import tempfile
import os
import subprocess
import platform
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch
import uuid
from bookme_scraper import scrape_all_bookme_data, format_all_data
from langchain.memory import ConversationBufferMemory


load_dotenv(override=True)

# === Setup ===
CHROMA_DIR = "./db/chroma_bert"
EMBED_MODEL = "bert-base-uncased"
GEMINI_MODEL = "gemini-1.5-flash"
BOOKME_CACHE = "bookme_scraper.txt"


def init_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))


def init_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def init_vector_store():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=init_embedding())


# === RAG Pipeline (Admin Interface) ===
def embed_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    db = init_vector_store()
    db.add_documents(docs)
    db.persist()
    return len(docs)


# === Tools ===
@tool
def get_dns_lookup(domain: str) -> str:
    """Useful for DNS lookup of a domain."""
    try:
        return socket.gethostbyname(domain)
    except socket.gaierror:
        return "Invalid domain."


@tool
def get_weather(city: str) -> str:
    """Useful for getting the weather of a city."""
    res = requests.get(f"https://wttr.in/{city}?format=3")
    return res.text if res.ok else "Unable to fetch weather."




@tool
def execute_command(command: str) -> str:
    """
    Executes shell commands or creates folders.
    Handles both direct commands (like 'ls -l') and natural language requests 
    (like 'make directory my_folder' or 'create folder test').
    """
    try:
        # Handle folder creation requests
        if any(cmd in command.lower() for cmd in ["mkdir", "make directory", "create folder"]):
            # Extract folder name from command
            folder_name = command.split(maxsplit=2)[-1].strip('"\'')
            
            # Create folder in current directory (safer than absolute paths)
            path = Path(folder_name)
            path.mkdir(exist_ok=True, parents=True)
            return f"Successfully created folder: {path.absolute()}"
        
        # Handle direct commands with platform awareness
        if platform.system() == "Windows":
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                text=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
        return result.stdout or "Command executed successfully"
        
    except subprocess.CalledProcessError as e:
        return f"Command failed (code {e.returncode}): {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def bookme_assistant(question: str) -> str:
    """
    Answers questions about Bookme.pk by using scraped data.
    It scrapes data only if not already available in cache.
    """
    def get_or_scrape_context():
        if os.path.exists(BOOKME_CACHE):
            with open(BOOKME_CACHE, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if cached:
                    return cached
        data = scrape_all_bookme_data()
        context = format_all_data(data)
        with open(BOOKME_CACHE, "w", encoding="utf-8") as f:
            f.write(context)
        return context

    try:
        context = get_or_scrape_context()
        llm = init_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following data scraped from Bookme.pk to answer the question:\n\n{context}"),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)
        result = document_chain.invoke({"input": question, "context": context})
        return result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        return f"Bookme tool error: {str(e)}"



@tool
def rag_tool(query: str) -> str:
    """
    Retrieval-Augmented Generation (RAG) tool.
    Uses BERT embeddings to search your uploaded and manually entered documents for relevant information and answers questions using retrieved context.
    """
    bert_embedding = init_embedding()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=bert_embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = init_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the following context: \n\n{context}."),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    try:
        result = rag_chain.invoke({"input": query})
        return result["answer"]
    except Exception as e:
        return f"RAG error: {str(e)}"


# Tool definitions
TOOLS = [
    get_dns_lookup,
    get_weather,
    execute_command,
    bookme_assistant,
    rag_tool,
]


# === Memory Handling ===
chat_history_file = "chat_history.json"


def save_chat(message):
    history = load_chat()
    history.append(message)
    with open(chat_history_file, "w") as f:
        json.dump(history, f)


def load_chat():
    if not os.path.exists(chat_history_file):
        return []
    with open(chat_history_file, "r") as f:
        return json.load(f)


# === Admin Streamlit Interface ===
def admin_interface():
    st.title("Admin Panel: RAG Manager")

    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if file and st.button("Embed and Save"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        chunks = embed_documents(tmp_path)
        st.success(f"Stored {chunks} document chunks using BERT.")

    # --- New: Add a text box for manual sentence embedding ---
    st.subheader("Add a Line and View BERT Embeddings")
    manual_line = st.text_input("Enter a line to embed and view tokens:")
    if manual_line:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        inputs = tokenizer(manual_line, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # shape: (num_tokens, 768)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        st.write(f"*Input line:* {manual_line}")
        for i, token in enumerate(tokens):
            st.write(f"*Token:* {token}")
            st.write(f"Embedding (first 5 values): {embeddings[i][:5].numpy()} ...")
            st.markdown("---")
        # Store the [CLS] embedding in ChromaDB
        bert_embedding = init_embedding()
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=bert_embedding)
        # Use a unique ID for each manual entry
        unique_id = str(uuid.uuid4())
        db.add_texts([manual_line], embeddings=[embeddings[0].numpy().tolist()], ids=[unique_id])
        st.success("Line and its embedding stored in vector database!")


# === User Streamlit Interface ===
def user_interface():
    st.title("User Chat Interface")
    query = st.text_input("Ask something...")

    if 'agent' not in st.session_state:
        llm = init_llm()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = initialize_agent(
            TOOLS,
            llm,
            agent_type="openai-tools",
            memory=memory,
            verbose=True
        )
        st.session_state['agent'] = agent
        st.session_state['memory'] = memory  # Optional: for debugging or manual access

    if query:
        save_chat({"user": query})
        agent = st.session_state['agent']
        modified_query = f"{query}\nIf no tool is useful, use the rag tool."

        try:
            result = agent.run(modified_query)
        except Exception as e:
            result = f"Failed to process the query: {e}"

        st.write("AI:", result)
        save_chat({"ai": result})


# === Streamlit Router ===
if __name__ == "__main__":
    role = st.sidebar.selectbox("Select Role", ["User", "Admin"])
    if role == "Admin":
        admin_interface()
    else:
        user_interface()