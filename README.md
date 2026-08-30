#  TravelAssist AI  
### RAG-Based Travel Support Chatbot

TravelAssist AI is a Retrieval-Augmented Generation (RAG) powered chatbot designed to answer travel-related queries using a structured FAQ knowledge base.

This project demonstrates how to build a production-style AI application by combining semantic search with large language models to deliver accurate, grounded responses.

---

##  Overview

Traditional chatbots often rely purely on model training data, which can lead to hallucinations and outdated responses.

TravelAssist AI solves this by:

- Retrieving relevant documents from a custom knowledge base
- Injecting retrieved context into the prompt
- Generating grounded responses using an instruction-tuned LLM
- Reducing hallucination using controlled prompt engineering

---

##  System Architecture


User Query
↓
Embedding Model (SentenceTransformers)
↓
Vector Similarity Search (FAISS)
↓
Top-K Relevant Documents
↓
Prompt Construction (Context Injection)
↓
FLAN-T5 LLM Generation
↓
Final Grounded Response


---

##  Technologies Used

### 🔹 Embeddings
- Model: `all-MiniLM-L6-v2`
- Converts text into dense semantic vectors
- Enables paraphrase-aware search

### 🔹 Vector Database
- FAISS (Facebook AI Similarity Search)
- Stores document embeddings
- Performs fast similarity search

### 🔹 LLM
- `google/flan-t5-base`
- Instruction-tuned
- Used for context-aware answer generation

### 🔹 Web Interface
- Streamlit
- Interactive chatbot UI
- Session-based conversation history

---


##  Project Structure

```bash
TravelAssist-AI/
│
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # CLI RAG system
├── chunking.py             # FAQ chunking logic
├── vector_store.py         # Embedding + FAISS creation
├── retriever.py            # Retrieval testing
│
├── data/
│   └── faq.txt             # Knowledge base
│
├── faiss_index.index       # Vector index (generated)
├── chunks.pkl              # Serialized chunks (generated)
│
├── requirements.txt
└── README.md
```

---

##  Key Features

- Semantic search using embeddings
- FAISS-based vector retrieval
- Top-K document grounding
- Hallucination-reduction prompt design
- Web-based interactive chatbot
- Conversation history tracking
- Fully local and deployable setup

---

##  Installation

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

▶️ Running the Application

**CLI Version**
```bash
python rag_pipeline.py
```

**Streamlit Web App**
```bash
streamlit run app.py
```

🌍 Deployment

This project can be deployed using:

- Streamlit Community Cloud
- HuggingFace Spaces
- Docker container
- AWS / GCP / Azure

 Evaluation Metrics

The system can be evaluated using:

- Retrieval Accuracy
- Relevant retrievals / Total queries

- Hallucination Rate
- Incorrect responses / Total responses

- Response Time
- Total response duration / Requests

- Escalation Rate
- Human handoffs / Total conversations

 Example Queries

- How long does refund take?
- Do I need a visa for Europe?
- What is baggage allowance?
- Can I receive refund as travel credit?

 Design Considerations

- Context-grounded prompting
- Controlled generation parameters
- Local vector storage
- Efficient chunking strategy
- Clean modular architecture



- End-to-end AI system design
- Practical RAG implementation
- LLM integration in production-style architecture
- Semantic search engineering
- AI-driven product development
