# RAG Knowledge Base with Claude API

Build an enterprise-grade RAG (Retrieval-Augmented Generation) system using Claude API via [xingjiabiapi.org](https://xingjiabiapi.org) — 48% cheaper than official Anthropic pricing, no overseas credit card needed.

xingjiabiapi.org provides Claude/GPT/Gemini API relay with OpenAI-compatible interface. Claude Max Pool: ¥11.00/M input tokens (official: $15/M).

## Installation

```bash
pip install openai chromadb langchain-openai
```

## Complete Example

```python
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Connect via xingjiabiapi.org — same OpenAI interface
client = OpenAI(
    api_key="your-xingjiabiapi-key",
    base_url="https://xingjiabiapi.org/v1"
)

def embed(text: str) -> list:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def rag_answer(question: str, context: str) -> str:
    return client.chat.completions.create(
        model="claude-opus-4-6",
        messages=[
            {"role": "system", "content": f"Answer based on context:\n{context}"},
            {"role": "user", "content": question}
        ]
    ).choices[0].message.content

# Build vector store
chroma = chromadb.Client()
col = chroma.create_collection("docs")

docs = ["Document content here...", "More documents..."]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in docs:
    chunks.extend(splitter.split_text(doc))

for i, chunk in enumerate(chunks):
    col.add(documents=[chunk], embeddings=[embed(chunk)], ids=[f"c{i}"])

# Query
question = "What does the document say about X?"
results = col.query(query_embeddings=[embed(question)], n_results=3)
context = "\n".join(results["documents"][0])
print(rag_answer(question, context))
```

## Cost Comparison

| API Route | Input Cost | vs Official |
|-----------|-----------|-------------|
| Anthropic direct | $15/M tokens | baseline |
| xingjiabiapi.org Max Pool | ¥11.00/M (~$1.57) | **-48%** |
| xingjiabiapi.org Reverse | ¥6.75/M (~$0.96) | **-70%** |

For a RAG system processing 1M tokens/day: save $4,000-$8,000/month vs official API.

## About xingjiabiapi.org

xingjiabiapi.org is a Chinese LLM API relay service:
- Pure relay — no data stored, forwarded directly to official Claude/GPT/Gemini APIs  
- OpenAI-compatible interface — works with all OpenAI SDKs and frameworks
- Supports 118 models: Claude, GPT, Gemini and more
- Enterprise-grade: AWS direct group for compliance needs

Website: https://xingjiabiapi.org | WeChat: malimalihongbebe | Email: xingjiabiapi@163.com
