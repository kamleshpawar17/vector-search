import glob
from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import hnswlib
import numpy as np
from loguru import logger
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import yaml

class DataBaseVectorSearch:
    def __init__(self, config_file_path: str = "./configs/config.yaml") -> None:
        with open(config_file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.model = SentenceTransformer(self.config['model_string'])
        self.vec_searcher = hnswlib.Index(space=self.config['space'], dim=self.config['dim'])
        self.vec_searcher.set_ef(self.config['ef'])
        self.database_image_index = []
        if Path(self.config['hnswlib_model_path']).exists():
            self.vec_searcher.load_index(self.config['hnswlib_model_path'])
            with open(self.config['database_image_index_path'], "rb") as file:
                self.database_image_index = pickle.load(file)
    
    def split_into_batches(self, fnames: list, batch_size: int) -> list:
        output = []
        for k in range(0, len(fnames), batch_size):
            if k+batch_size < len(fnames):
                output.append(fnames[k: k+batch_size])
            else:
                output.append(fnames[k::])
        return output
        
    def add_images_from_dir(self, dir_path: str, batch_size: int = 5000) -> None:
        # generate embedding for all the images
        fnames = glob.glob(dir_path)
        if (len(fnames)) > batch_size:
            fnames_batched = self.split_into_batches(fnames, batch_size)
        else:
            fnames_batched = []
            fnames_batched = fnames_batched.append(fnames)
        logger.info(f"computing embeddings...")
        for fnames in fnames_batched:
            embs = []
            for fname in tqdm(fnames):
                emb = self.model.encode(Image.open(fname))
                embs.append(emb)
            embs = np.array(embs)
            assert len(embs) == len(fnames), "Number of embeddings not equal to number of images in directory"
            self.database_image_index += fnames
            
            # add embedding to the database
            logger.info("updating hnswlib model....")
            if not Path(self.config['hnswlib_model_path']).exists():
                database_length = 0
                self.vec_searcher.init_index(max_elements=len(fnames), ef_construction=self.config['ef_construction'], M=self.config['m'])
            else:
                database_length = self.vec_searcher.get_current_count()
                self.vec_searcher.resize_index(database_length+len(fnames))
            # put data for indexing
            self.vec_searcher.add_items(embs, np.arange(database_length, database_length+len(embs)))
            # save model and database
            logger.info(f"saving hnswlib model to {self.config['hnswlib_model_path']}")
            self.vec_searcher.save_index(self.config['hnswlib_model_path'])
            with open(self.config['database_image_index_path'], "wb") as file:
                pickle.dump(self.database_image_index, file)
    
    def query_database_prompt(self, prompt: str, number_of_images: int = 10, show_images: bool = True) -> list:
        # knn search using hnswlib model
        emb = self.model.encode(prompt)
        neighbors, scores = self.vec_searcher.knn_query(
                emb, k=number_of_images
            )
        neighbors, scores = np.squeeze(neighbors).astype('int'), np.squeeze(scores)
        if show_images:
            self.show_images(neighbors, scores)
        return [self.database_image_index[k] for k in neighbors]
    
    def query_database_image(self, image_path: str, number_of_images: int = 10, show_images: bool = True) -> None:
        # knn search using hnswlib model
        emb = self.model.encode(Image.open(image_path))
        neighbors, scores = self.vec_searcher.knn_query(
                emb, k=number_of_images
            )
        neighbors, scores = np.squeeze(neighbors).astype('int'), np.squeeze(scores)
        if show_images:
            self.show_images(neighbors, scores, image_path)
        return [self.database_image_index[k] for k in neighbors]
    
    def show_images(self, image_indices: list, scores: list, query_image_path: str = None):
        """function to display the input and matched images

        Args:
            image_indices (list): list of matched/searched image index
            scores (List[str]): score for the matched images
            query_image_path (str): query image path
        """
        number_of_images = len(image_indices)
        cols = 5
        rows = np.ceil(number_of_images / cols) + 1
        plt.figure(figsize=(12, 8))
        start = 0
        if query_image_path is not None:
            plt.subplot(rows, cols, 1)
            frame = cv2.imread(query_image_path)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Query image")
            plt.axis("off")
            start = 1
        for k in range(start, number_of_images):
            plt.subplot(rows, cols, k + 1)
            frame = cv2.imread(self.database_image_index[image_indices[k]])
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"score: {1.0-scores[k]:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        
        
        
        
        
            