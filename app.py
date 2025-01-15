import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# Initialize Sentence Transformers for embeddings (free and open source)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# Create a custom embedding function using Sentence Transformers
class SentenceTransformerEmbedding(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return embedding_model.encode(texts).tolist()

# Initialize collection with the custom embedding function
embedding_func = SentenceTransformerEmbedding()
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_func
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load and process documents
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Function to generate embeddings using Sentence Transformers
def get_embedding(text):
    print("==== Generating embeddings... ====")
    return embedding_model.encode(text).tolist()

# Generate embeddings and upsert documents
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_embedding(doc["text"])
    print("==== Inserting chunks into db... ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    relevant_chunks = [doc for doc in results["documents"][0]]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Function to generate a response using Groq
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Using Llama2-70B as it's available on Groq
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message

# Example usage
if __name__ == "__main__":
    question = "tell me about Angshuman"
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer.content)