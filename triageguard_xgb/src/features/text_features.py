import pandas as pd
import numpy as np
import os
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class TextFeatureExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_components: int = 8, pca_path: str = None):
        self.model_name = model_name
        self.n_components = n_components
        self.pca_path = pca_path
        
        # Lazy loading of the embedding model
        self.embedder = None
        
        # PCA object
        self.pca = None
        if self.pca_path and os.path.exists(self.pca_path):
            self.pca = joblib.load(self.pca_path)
            
    def _load_embedder(self):
        if self.embedder is None:
            print(f"Loading embedding model {self.model_name}...")
            self.embedder = SentenceTransformer(self.model_name)
            
    def extract_features(self, text_series: pd.Series, is_training: bool = False) -> pd.DataFrame:
        """
        Embeds the text and applies PCA.
        If is_training is True, fits PCA and saves it if pca_path is set.
        """
        self._load_embedder()
        
        # Replace NaNs with empty string
        texts = text_series.fillna("").astype(str).tolist()
        
        print(f"Encoding {len(texts)} text complaints...")
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        if is_training:
            print(f"Fitting PCA with {self.n_components} components...")
            self.pca = PCA(n_components=self.n_components)
            pca_features = self.pca.fit_transform(embeddings)
            
            if self.pca_path:
                os.makedirs(os.path.dirname(self.pca_path), exist_ok=True)
                joblib.dump(self.pca, self.pca_path)
                print(f"Saved PCA to {self.pca_path}")
        else:
            if self.pca is None:
                raise ValueError("PCA is not fitted and no pca_path was loaded.")
            pca_features = self.pca.transform(embeddings)
            
        # Create a DataFrame for the PCA components
        pca_cols = [f'text_pca_{i}' for i in range(self.n_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=text_series.index)
        
        return pca_df
