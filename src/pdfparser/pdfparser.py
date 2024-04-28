import os
from typing import Union

import easyocr
import fitz
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import pipeline
from PIL import Image

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
            pickler(filename=self.collection_name+'.pickle', data=processed_docs)

        return processed_docs

## Embed to VectorStore

class NeuralSearcher:
    """
    Quick-and-dirty PDF search using embeddings.
    
    Args:
        collection_name (str): name of doc collection for qdrant
        recreate (bool): Whether to recreate the collection or not. Defaults to False. 
        documents (list, str, None): documents to add to the vector store
        data_dir (str): path to data directory with pdf images
    
    Usage:
    ```
    from pdfparser import NeuralSearcher
    
    # create embeddings from parsed documents (pickle file)
    searcher = pp.NeuralSearcher('student_manual', recreate=True, documents='student_manual.pickle')

    # retrieve pages 
    query = 'How do the checkpoints help in case of failure? How can ML teams benefit from HPE MLDE?'
    searcher.query(query)
    
    ```
    """
    def __init__(self, collection_name:str, recreate:bool=False, documents:Union[None, list, str]=None, data_dir:str='data/imgs'):
        self.collection_name = collection_name
        self.data_dir = data_dir
        
        # Initialize encoder model
        self.encoder = load("BAAI/bge-small-en-v1.5")
        
        self.instruction = 'Represent this sentence for searching relevant passages: '

        # Initialize the q&a pipeline
        qa_model = "deepset/tinyroberta-squad2"
        self.pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_model, cache_folder='models')

        # initialize Qdrant client
        qdrant_path = os.path.join('data', 'qdrant')
        os.makedirs(qdrant_path, exist_ok=True)
        self.qdrant = QdrantClient(path=qdrant_path)
        
        if recreate:
            self.create_collection()
            
        if isinstance(documents, str):
            documents = unpickler(documents)
            self.load_documents(documents)
        
        elif isinstance(documents, list):
            self.load_documents(documents)
 
    def create_collection(self):
        # Create collection
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),  # vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )

    def load_documents(self, documents):
        # upload documents to the collection
        self.qdrant.upload_records(
            collection_name=self.collection_name,
            records=[
                models.Record(
                    id=idx, vector=self.encoder.encode(doc["text"], normalize_embeddings=True).tolist(), payload=doc
                )
                for idx, doc in tqdm(enumerate(documents))
            ],
        )

    def show_results(self,query,hits, verbose):
        output_str = ''
        output_str += f'Query: {query}\n'
        for hit in hits:
            if verbose==1:
                output_str += f"Page: {hit.payload['page']+1} | Score: {hit.score:.3f}\n"

            if verbose==2:
                output_str += f"[Page {hit.payload['page']+1}] '{hit.payload['text'].splitlines(True)[0]}' (score: {hit.score:.3f})\n"

            elif verbose>2:
                output_str += f">Page: {hit.payload['page']+1} (score: {hit.score:.3f})\n"
                output_str += f">>Content:{hit.payload['text']}\n"
                show_imgs(hit.payload['page'], self.data_dir) 
        
        print(output_str)

        return output_str

    # Ask the engine a question:
    def query(self, query:str, limit:int=3, with_vectors:bool=False, verbose:int=1) -> list:
        # Encode question
        query_vector = self.encoder.encode(self.instruction+query, normalize_embeddings=True).tolist()
        
        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            with_vectors=with_vectors,
            limit=limit,
        )
        
        output_str = ''
        # Display
        if verbose>=1:
            output_str = self.show_results(query, hits, verbose)
            
        if with_vectors:
            return hits, query_vector
        
        return hits, output_str

    def extract(self, query:str, results:list, limit:int=3, verbose:int=1) -> list:
        if verbose>=1:
            print(f'Query: {query}\n')

        # get context from results
        outputs = []
        text_outputs = str()

        for i in range(limit):
            page = results[i].payload['page']+1
            context = results[i].payload['text']
            
            input = {'question': query,'context': context }
            output = self.pipeline(input)
            outputs.append(output)
            
            if verbose==1:
                print(f">[page {page}] Quick answer: {output['answer'].strip()} (score: {output['score']:.3f})")
                text_outputs.append(f"Page {page}: {output['answer'].strip()} (score: {output['score']:.3f})\n")
            elif verbose==2:
                print(f"\n>[page {page}: \"{context.splitlines()[0]}\"]\nQuick answer: {output['answer'].strip()} (score: {output['score']:.3f})")
            elif verbose>2:
                print(f"\n>[page {page}: \"{context.splitlines()[0]}\" (score: {results[i].score:.3f})]\nQuick answer: {output['answer'].strip()} (score: {output['score']:.3f})")


        return outputs, text_outputs

    def quick_search(self, query:str, limit:int=5, verbose:tuple=(2,0)):
        print('---------------------\nQuerying documents...\n---------------------') if verbose[0]>0 else None 
        results, output_str = self.query(query=query, limit=limit, verbose=verbose[0])
        print('---------------------\nExtracting information...\n---------------------') if verbose[1]>0 else None 
        _, text_outputs = self.extract(query=query, results=results, limit=limit, verbose=verbose[1])
        return output_str
    
    def image_search(self, filename, limit:int=5, verbose:tuple=(2,0)): # MODIFY FOR IMAGE

        #img = Image.fromarray(image.astype('uint8'), 'RGB')
        # filename = "/tmp/window_title_image.jpg"
        #img.save(filename)

        ocr_text = reader.readtext(filename, detail=0)
        print('Detected text:', ocr_text) if verbose[0]>0 else None

        # Combine the OCR text into a single query string
        query = ' '.join(ocr_text)
        print('---------------------\nQuerying documents...\n---------------------') if verbose[0]>0 else None 
        results, output_str = self.query(query=query, limit=limit, verbose=verbose[0])
        print('---------------------\nExtracting information...\n---------------------') if verbose[1]>0 else None 
        _, text_outputs = self.extract(query=query, results=results, limit=limit, verbose=verbose[1])
        return output_str