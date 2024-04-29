import os
from typing import Union

import easyocr
import fitz
from PIL import Image
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import pipeline

from .utils import add_keywords, load, pickler, show_imgs, unpickler

reader = easyocr.Reader(['en'])


class PdfProcessor:
    def __init__(self, collection_name:str, pdf:str, data_dir:str='data/imgs', auto:bool=False):
        """PDF Processor. Extracts text and images.

        Args:
            collection_name (str): Name of resulting vector store collection
            pdf (str): Filepath for input pdf
            auto (bool, optional): Processes documents automatically. Defaults to False.
        """
        self.collection_name = collection_name
        self.pdf = pdf
        self.data_dir = data_dir
        
        os.makedirs(self.data_dir, exist_ok=True)

        if auto:
            _ = self.document_processor()
        
        
    # Parse through the PDF
    def process_image(self, img_bytes, ext, page, n) -> str:
            pix = fitz.Pixmap(img_bytes)
            filename = os.path.join(self.data_dir, f"{page}_{n}.{ext}")
            pix.save(filename)
            ocr_text = reader.readtext(filename, detail=0)
            return ' '.join(ocr_text)

    def extract_content(self,page) -> str:
        # Extract text
        textpage = page.get_textpage().extractBLOCKS()
        textpage_data =  [(x0, y0, x1, y1, content.strip(), block_no, block_type) 
                        for (x0, y0, x1, y1, content, block_no, block_type) in textpage]
        textpage_sorted = sorted(textpage_data, key=lambda text: (text[1], text[0]))
        text = '\n'.join([content[4] for content in textpage_sorted])

        # Extract images
        # get_images -> (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
        # extract_image -> {ext, smask, width, height, colorspace, cs-name, xres, yres, image}
        imgs_extracted = [ self.doc.extract_image(data[0]) for data in page.get_images() ]
        if imgs_extracted != []:
            img_ocrs = [ self.process_image(img['image'], img['ext'], page.number, i) for i, img in enumerate(imgs_extracted) ]
            keywords = [ add_keywords(img) for img in img_ocrs]
            text = text+'\n'.join(keywords)
        
        return text
        
    def document_processor(self, pickle:bool=True) -> list:
        self.doc = fitz.open(self.pdf) # open a document
        processed_docs = [ { 'page': page.number, 'text': self.extract_content(page)} for page in tqdm(self.doc) ]
        print(f'Document {self.pdf} was processed correctly')
        
        if pickle:
            pickler(filename='data/'+self.collection_name+'.pickle', data=processed_docs)

        return processed_docs