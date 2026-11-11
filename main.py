from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
with open("data/company.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(text)
chunks = text.split("\n\n")

print(chunks)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print(embeddings)
client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="company_docs",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
points = []

for i, embedding in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunks[i]
            }
        )
    )
client.upsert(
    collection_name="company_docs",
    points=points
)
question = input("Ask a question: ")
question_embedding = model.encode(question)
search_result = client.query_points(
    collection_name="company_docs",
    query=question_embedding.tolist(),
    limit=3,
    with_payload=True
).points
context = ""

for result in search_result:
    context += result.payload["text"] + "\n"
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
    {
    "role": "system",
    "content": """Answer the question using ONLY the information in the provided context.
Do not add information from your own knowledge.
If the answer is not present in the context, say:
"I don't know based on the provided documents."
"""
},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]
)
answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)
