import gradio as gr
import src.pdfparser as pp
import pygetwindow as gw
import pyautogui

searcher = pp.NeuralSearcher('student_manual')

def search(query):
    return searcher.quick_search(query, verbose=(2,0))

def image_search(image):
    return searcher.image_search(image, verbose=(2,0))

demo = gr.Interface(
    fn=image_search,
    inputs=["image"],
    outputs=["text"]
)

demo.launch()
