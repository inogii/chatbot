import os
from typing import Union
import subprocess
import json
import easyocr
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import pipeline
import requests

from .utils import add_keywords, load, pickler, show_imgs, unpickler
from .instructions import INSTRUCTIONS

reader = easyocr.Reader(['en'])

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
    

    def generate_response(self, prompt):
        url = "http://localhost:11434/api/generate"
        data = {"model": "mistral", "prompt": prompt}
        
        try:
            response = requests.post(url, json=data)
            # Instead of using response.json(), read the raw response text
            # and manually process each line or JSON object
            full_response = ""
            for line in response.iter_lines():
                if line:  # ignore empty lines
                    try:
                        response_data = json.loads(line.decode('utf-8'))
                        full_response += response_data.get("response", "")
                    except json.JSONDecodeError as e:
                        return f"Invalid JSON format in response: {e}", 500
            return full_response
        except requests.RequestException as e:
            return f"Request failed: {e}", 500


    def extract(self, query:str, prompt:str, results:list, limit:int=3, verbose:int=1) -> list:
        if verbose>=1:
            print(f'Query: {query}\n')

        # get context from results
        # outputs = []
        text_outputs = str()

        context = ''

        for i in range(limit):
            #page = results[i].payload['page']+1
            context += results[i].payload['text']
            
            # input = {'question': query,'context': context }
            # output = self.pipeline(input)
            # outputs.append(output)

        if prompt is None:
            prompt = INSTRUCTIONS
        
        prompt = f"{prompt} \n Question: {query} \n Context: {context}"

        output = self.generate_response(prompt)

        if verbose==1:
            text_outputs += str(output)
            pass


        return output, text_outputs

    def quick_search(self, query:str, limit:int=2, verbose:tuple=(2,0)):
        print('---------------------\nQuerying documents...\n---------------------') if verbose[0]>0 else None 
        results, output_str = self.query(query=query, limit=limit, verbose=verbose[0])
        print('---------------------\nExtracting information...\n---------------------') if verbose[1]>0 else None 
        outputs, text_outputs = self.extract(query=query, prompt=None, results=results, limit=limit, verbose=verbose[1])
        return outputs
    
    def image_search(self, filename, prompt, limit:int=3, verbose:tuple=(2,0)): # MODIFY FOR IMAGE

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
        _, text_outputs = self.extract(query=query, prompt=prompt, results=results, limit=limit, verbose=verbose[1])
        return text_outputs