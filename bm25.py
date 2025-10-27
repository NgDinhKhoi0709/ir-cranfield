import json
import pickle
import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class BM25Search:
    def __init__(self, preprocessed_path="preprocessed_cranfield.json", checkpoint_path="bm25_checkpoint.pkl"):
        self.preprocessed_path = preprocessed_path
        self.checkpoint_path = checkpoint_path
        
        # Try to load from checkpoint first
        if os.path.exists(checkpoint_path):
            print(f"📁 Loading BM25 model from checkpoint: {checkpoint_path}")
            self.load_checkpoint()
        else:
            print(f"🔄 Training new BM25 model...")
            # Load data
            with open(preprocessed_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            
            # Chuẩn bị corpus
            self.corpus = [d["terms"] for d in self.docs]
            self.doc_ids = [d["doc_id"] for d in self.docs]

            # Train BM25
            self.bm25 = BM25Okapi(self.corpus)
            print(f"✅ BM25 model initialized with {len(self.corpus)} documents")
            
            # Save checkpoint
            self.save_checkpoint()
            print(f"💾 Model saved to checkpoint: {checkpoint_path}")

    def save_checkpoint(self):
        """Save BM25 model and related data to checkpoint file"""
        checkpoint_data = {
            'bm25': self.bm25,
            'corpus': self.corpus,
            'doc_ids': self.doc_ids,
            'docs': self.docs
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self):
        """Load BM25 model and related data from checkpoint file"""
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.bm25 = checkpoint_data['bm25']
        self.corpus = checkpoint_data['corpus']
        self.doc_ids = checkpoint_data['doc_ids']
        self.docs = checkpoint_data['docs']
        print(f"✅ BM25 model loaded from checkpoint with {len(self.corpus)} documents")
    
    def search(self, query, top_k=10):
        query_tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(list(zip(self.doc_ids, scores)), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    # First run will train and save checkpoint
    print("=== First initialization ===")
    model = BM25Search("dataset/preprocessed_cranfield.json")
    
    query = "aerodynamic flow in jet stream"
    results = model.search(query)
    print("\n🔎 Top results:")
    for doc_id, score in results:
        print(f"Doc {doc_id}: {score:.4f}")
    
    # Second run will load from checkpoint (much faster)
    print("\n=== Second initialization (from checkpoint) ===")
    model2 = BM25Search("dataset/preprocessed_cranfield.json")
    
    # Test same query
    results2 = model2.search(query)
    print("\n🔎 Top results (from checkpoint):")
    for doc_id, score in results2:
        print(f"Doc {doc_id}: {score:.4f}")
