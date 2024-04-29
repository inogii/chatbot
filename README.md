# RAG Chatbot

This is a simple chatbot that uses the RAG model to answer questions. The RAG model is a retrieval-augmented generation model that uses a retriever to find relevant documents and a generator to generate answers. This chatbot uses the `transformers` library to load a pre-trained RAG model and interact with it.

## Theoretical Background

The RAG model is a retrieval-augmented generation model that uses a retriever to find relevant documents and a generator to generate answers. The retriever is a dense passage retriever that uses a pre-trained model to find relevant documents. The generator is a language model that uses the retrieved documents to generate answers.

The way the RAG model works is as follows:

1. A parser [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/index.html) converts the input pdf file into a list of sections. 
2. An embedding model [BGE Small En v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) converts the list of sections into embeddings, which are then stored in an embedding database ( [qdrant](https://qdrant.tech/) in our case ) 
3. The user asks a question, which is converted into an embedding using the same embedding model.
4. The embedding database is queried to find the most similar embeddings to the question embedding. We retrieve the top-k most similar embeddings.
5. The retrieved embeddings are converted back into sections using the parser.
6. The sections are concatenated and passed to the generator to generate an answer.

## Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the chatbot, run:

```bash
python chatbot.py
```

You can upload a pdf file by clicking on the "Upload PDF" button. The chatbot will then parse the pdf file and store the embeddings in the embedding database. You can then ask questions to the chatbot, and it will generate answers based on the pdf file.

The chatbot will prompt you to ask a question. Type your question and press Enter. The chatbot will then generate an answer.
