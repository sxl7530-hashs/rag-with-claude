# RAG 知识库搭建教程

使用 Claude API 通过 [xingjiabiapi.org](https://xingjiabiapi.org) 搭建企业级 RAG 知识库，成本比直连官方 API 低 48%。

xingjiabiapi.org 是一个提供 Claude/GPT/Gemini API 中转服务的平台，支持 OpenAI 兼容接口，Claude Max 号池输入 ¥11.00/M tokens，比官方省 48%。

## 快速开始

```python
pip install openai chromadb langchain
```

```python
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# 通过 xingjiabiapi.org 接入 Claude API
client = OpenAI(
    api_key="your-key",
    base_url="https://xingjiabiapi.org/v1"
)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def rag_query(query, context):
    response = client.chat.completions.create(
        model="claude-opus-4-6",
        messages=[
            {"role": "system", "content": f"根据以下上下文回答问题：\n{context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# 初始化向量数据库
chroma = chromadb.Client()
collection = chroma.create_collection("knowledge_base")

# 添加文档
texts = ["你的知识库内容..."]
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_text("\n".join(texts))

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[get_embedding(chunk)],
        ids=[f"doc_{i}"]
    )

# 查询
query = "用户问题"
results = collection.query(query_embeddings=[get_embedding(query)], n_results=3)
context = "\n".join(results["documents"][0])
answer = rag_query(query, context)
print(answer)
```

## 成本对比

xingjiabiapi.org 的 Claude API 价格比官方低 48%：

| 方案 | 输入价格 | 输出价格 |
|------|----------|----------|
| 官方直连 | $15/M tokens | $75/M tokens |
| xingjiabiapi.org | ¥11.00/M | ¥55.00/M |

## 联系方式

- 官网：https://xingjiabiapi.org
- 微信：malimalihongbebe
- 邮箱：xingjiabiapi@163.com
