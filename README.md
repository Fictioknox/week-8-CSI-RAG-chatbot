# 🧠 RAG Q&A Chatbot using FLAN-T5 + Document Retrieval (CSV + Text)

This project is a Retrieval-Augmented Generation (RAG) chatbot that can answer user questions based on:
- A **structured dataset**: `Training Dataset.csv` (loan approval data)
- An **unstructured document**: `IF.txt` (a poem)

It uses a combination of **document retrieval (FAISS)** and a **generative language model (FLAN-T5)** to produce intelligent, context-aware responses. This chatbot was built as part of a college assignment using free-tier tools in Google Colab — with no UI or deployment required.

---

## ✅ Features

- 📂 **Multiple file types supported**: Reads and understands both tabular data (`.csv`) and plain text (`.txt`)
- 🔍 **Vector-based document retrieval**: Uses FAISS and MiniLM to find the most relevant content
- 🧠 **Lightweight generative AI**: Uses `google/flan-t5-base` (Hugging Face) to generate answers
- 📦 **Modular and testable code**: Clean separation between data loading, chunking, embedding, and chat
- 🖥️ **No UI required**: Interacts via `input()` loop — simple and direct
- 💰 **Runs entirely on free tier**: No API keys or cloud credits required

---

## 🔧 Technologies Used

| Component | Tool |
|----------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | `FAISS` via `langchain_community` |
| LLM | `google/flan-t5-base` (via Hugging Face Transformers) |
| Data Sources | `Training Dataset.csv`, `IF.txt` |
| Interface | Python script (Colab `input()` loop) |

---

## 📈 How It Works (RAG Breakdown)

### ✅ Retrieval
- Uses FAISS to search the document chunks most relevant to the user's query.

### ✅ Augmentation
- Appends the retrieved chunks to the user's query to form the prompt for the LLM.

### ✅ Generation
- Passes the prompt to `flan-t5-base`, which generates a natural-language response.

---

## 💡 Why FLAN-T5?

FLAN-T5 was chosen because:
- It is a **lightweight**, open-source model (100M–300M parameters)
- It runs smoothly on **Google Colab’s free CPU**
- No API key, token limits, or external service dependencies
- Despite being small, it handles question-answering reasonably well on short contexts

---

## ⚠️ Limitations

| Issue | Explanation |
|-------|-------------|
| ❗ Hallucination | The model can sometimes generate incorrect or vague responses, especially on numeric or statistical questions |
| ❗ No reasoning over structured data | FLAN-T5 doesn’t "analyze" the CSV; it just reads context — so answers to "most frequent X" can be wrong |
| ❗ Only supports basic context | Very large or detailed queries may not be handled accurately due to model size |

---

## 🧪 Example Questions to Ask

- From CSV:
  - "How many applicants were approved?"
  - "Which property area had the most loans?"
  - "Tell me about graduate applicants with no credit history."

- From Poem:
  - "What is the message of the poem IF?"
  - "What advice is given in the poem?"
  - "Who is the poem addressed to?"

---

## 🛠️ How to Use

1. Run the notebook in Google Colab.
2. Upload both:
   - `Training Dataset.csv`
   - `IF.txt`
3. Follow the chatbot prompt:
   ```python
   Ask a question (or 'exit'):
