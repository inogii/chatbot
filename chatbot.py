import warnings
import json
import pickle

import gradio as gr

import src.neuralsearcher as ns
import src.pdfparser as pp

warnings.filterwarnings('ignore', category=UserWarning)

searcher = None

def process_pdf(file_path):
    global searcher

    pp.PdfProcessor(collection_name='rag_queries', pdf=file_path, auto=True)
    searcher = ns.NeuralSearcher('rag_queries', recreate=True, documents='data/rag_queries.pickle')

    jsonfile = pickle.load(open("data/rag_queries.pickle", "rb"))
    json.dump(jsonfile, open("data/rag.json", "w"), indent=4)

    return "PDF uploaded and processed. Ready for RAG queries."

def ask_llm(question):
    global searcher
    print("Asking LLM...")
    response = searcher.quick_search(question, verbose=(2,0))
    print(f"Response: {response}")
    return response

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Chatbot Interface")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Upload PDF for RAG")
                file_input = gr.File(label="Upload your PDF file")
                submit_button = gr.Button("Process PDF")
                output_pdf = gr.Label()
                submit_button.click(process_pdf, inputs=file_input, outputs=output_pdf)
            
            with gr.Column():
                gr.Markdown("#### Chat with LLM")
                text_input = gr.Textbox(label="Enter your question")
                submit_query = gr.Button("Ask")
                output_text = gr.Textbox(label="LLM Response", interactive=False)
                submit_query.click(ask_llm, inputs=text_input, outputs=output_text)

    demo.launch()

if __name__ == "__main__":
    main()
