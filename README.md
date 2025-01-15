# A POC on Retrieval-Augmented Generation.
# An inhouse Question Answer Engine based on LLM

This POC implements an inhouse question-answering system using Groq's LLM API and local document embeddings. It allows users to query a collection of documents and receive AI-generated responses based on the relevant content.

## Features

- Document loading and processing from local directory
- Text chunking with configurable size and overlap
- Document embedding using Sentence Transformers
- Vector storage using ChromaDB
- Question answering using Groq's Llama 3.3 70B model
- Persistent storage of embeddings

## Prerequisites

- Python 3.8 or higher
- Groq API key
- Local directory containing text documents to process

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install groq sentence-transformers chromadb python-dotenv
```

3. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Create a directory named `news_articles` in the project root and add your text documents there.

## Project Structure

```
project_root/
│
├── main.py            # Main application file
├── .env              # Environment variables
├── news_articles/    # Directory for text documents
│   ├── doc1.txt
│   └── doc2.txt
└── chroma_persistent_storage/  # Vector database storage (created automatically)
```

## Usage

1. Place your text documents in the `news_articles` directory.

2. Run the script:
```bash
python main.py
```

3. The script will:
   - Load documents from the specified directory
   - Split them into manageable chunks
   - Generate embeddings using Sentence Transformers
   - Store the embeddings in ChromaDB
   - Process queries and generate responses using Groq's LLM

4. To ask questions, modify the `question` variable in the main block:
```python
if __name__ == "__main__":
    question = "your question here"
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer.content)
```

## Configuration Options

You can modify the following parameters in the code:

- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 20 characters)
- `n_results`: Number of relevant chunks to retrieve (default: 2)
- `temperature`: LLM temperature setting (default: 0.7)
- `max_tokens`: Maximum tokens in response (default: 150)

## Key Components

1. **Document Loading**: `load_documents_from_directory()`
   - Loads text files from the specified directory
   - Supports .txt files

2. **Text Chunking**: `split_text()`
   - Splits documents into smaller chunks for processing
   - Implements overlap to maintain context

3. **Embedding Generation**: `SentenceTransformerEmbedding`
   - Uses the 'all-MiniLM-L6-v2' model
   - Converts text chunks into vector embeddings

4. **Vector Storage**: ChromaDB
   - Persistent storage of document embeddings
   - Efficient similarity search

5. **Query Processing**: `query_documents()`
   - Retrieves relevant document chunks based on query
   - Uses vector similarity search

6. **Response Generation**: `generate_response()`
   - Utilizes Groq's Llama 3.3 70B model
   - Generates concise, contextual responses

## Important Notes

- The script sets `TOKENIZERS_PARALLELISM=false` to prevent potential deadlocks
- Responses are limited to three sentences for conciseness
- The system uses persistent storage, so embeddings are preserved between runs

## Troubleshooting

1. If you encounter memory issues:
   - Reduce the chunk_size
   - Process fewer documents at once

2. If responses are not relevant:
   - Increase the n_results parameter
   - Adjust the chunk_size and overlap

3. For tokenizer warnings:
   - Ensure the `TOKENIZERS_PARALLELISM` environment variable is set correctly

## License

MIT License

Copyright (c) 2024 Angshuman Nandi.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

First off, thank you for considering contributing to this project! It's people like you that make it a great tool for everyone.
Code of Conduct
By participating in this project, you are expected to uphold our Code of Conduct:

Use welcoming and inclusive language.
Be respectful of different viewpoints and experiences.
Gracefully accept constructive criticism.
Focus on what is best for the community.
Show empathy towards other community members.
