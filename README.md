# Q&A Helper

### Introduction

This Q&A helper facilitates document processing, embedding creation, and efficient searching within PDF documents. It utilizes EasyOCR for text extraction, Fitz for PDF parsing, and Qdrant for vector storage and searching.

### Dependencies

    easyocr
    fitz
    qdrant_client
    tqdm
    PyMuPDF
    sentence_transformers
    gradio
    pygetwindow
    pyautogui

### Usage

PDF Processor (PdfProcessor):
- PdfProcessor extracts text and images from PDFs.
- Initialize with collection_name, pdf, and optionally data_dir and auto.
- Call document_processor() to process the PDF. Optionally, pickle the results.

Neural Searcher (NeuralSearcher):
- NeuralSearcher creates embeddings from parsed documents and enables quick PDF search.
- Initialize with collection_name, recreate, documents, and data_dir.
- Call query() with a query string to retrieve relevant pages from the documents.

### Example

```
python

from pdfparser import NeuralSearcher, PdfProcessor

# PDF Processor
pdf_processor = PdfProcessor(collection_name='documents', pdf='example.pdf', auto=True)

# Neural Searcher
searcher = NeuralSearcher(collection_name='documents', recreate=True, documents='example.pickle')

# Query
query = 'How do the checkpoints help in case of failure?'
searcher.query(query)
```

### Notes

Ensure the required models are downloaded and accessible for proper functioning.


### Roadmap
- Currently using `BAAI/bge-small-en-v1.5` embedding model. Consider alternatives.
- Currently using `deepset/tinyroberta-squad2`for the q&a pipeline. This doesn't work. Consider other llms: `mixtral-7b` or `mistral`.
- Implement advanced RAG techniques. For example, creating multiple queries from the original one before the vector similarity search, then reranking. 
- Add multi-modal input. Include an ocr to extract the question (and multiple anwers) from an input image.
- GUI. Consider `chainlit`for a chatbot-like interface. 