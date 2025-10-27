import json
import os
import math
from collections import defaultdict
from bm25 import BM25Search

class IREvaluator:
    def __init__(self, query_file="dataset/cranfield/query.txt", 
                 relevance_dir="dataset/cranfield/res/",
                 preprocessed_path="dataset/preprocessed_cranfield.json"):
        """
        Initialize IR Evaluator
        
        Args:
            query_file: Path to query file
            relevance_dir: Directory containing relevance judgment files
            preprocessed_path: Path to preprocessed documents
        """
        self.query_file = query_file
        self.relevance_dir = relevance_dir
        self.preprocessed_path = preprocessed_path
        
        # Load queries and relevance judgments
        self.queries = self.load_queries()
        self.relevance_judgments = self.load_relevance_judgments()
        
        # Initialize BM25 model
        self.bm25_model = BM25Search(preprocessed_path)
        
        print(f"✅ Loaded {len(self.queries)} queries")
        print(f"✅ Loaded relevance judgments for {len(self.relevance_judgments)} queries")
    
    def load_queries(self):
        """Load queries from query.txt file"""
        queries = {}
        with open(self.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        query_id = int(parts[0])
                        query_text = parts[1]
                        queries[query_id] = query_text
        return queries
    
    def load_relevance_judgments(self):
        """Load relevance judgments from res/ directory"""
        relevance_judgments = {}
        
        for filename in os.listdir(self.relevance_dir):
            if filename.endswith('.txt'):
                query_id = int(filename.replace('.txt', ''))
                relevance_judgments[query_id] = {}
                
                filepath = os.path.join(self.relevance_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 3:
                                qid = int(parts[0])
                                doc_id = int(parts[1])
                                relevance = int(parts[2])
                                
                                # Only store positive relevance scores
                                if relevance > 0:
                                    relevance_judgments[query_id][doc_id] = relevance
        
        return relevance_judgments
    
    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc_id, _ in retrieved_k if doc_id in relevant_docs)
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Recall@K"""
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc_id, _ in retrieved_k if doc_id in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_average_precision(self, retrieved_docs, relevant_docs):
        """Calculate Average Precision (AP) for a single query"""
        if len(relevant_docs) == 0:
            return 0.0
        
        ap = 0.0
        relevant_retrieved = 0
        
        for i, (doc_id, score) in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                ap += precision_at_i
        
        return ap / len(relevant_docs)
    
    def calculate_reciprocal_rank(self, retrieved_docs, relevant_docs):
        """Calculate Reciprocal Rank (RR) for a single query"""
        for i, (doc_id, score) in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_dcg_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Discounted Cumulative Gain (DCG) at rank k"""
        dcg = 0.0
        for i, (doc_id, score) in enumerate(retrieved_docs[:k]):
            if doc_id in relevant_docs:
                relevance = relevant_docs[doc_id]
                if i == 0:
                    dcg += relevance
                else:
                    dcg += relevance / math.log2(i + 1)
        return dcg
    
    def calculate_ideal_dcg_at_k(self, relevant_docs, k):
        """Calculate Ideal DCG (IDCG) at rank k"""
        # Sort relevance scores in descending order
        sorted_relevances = sorted(relevant_docs.values(), reverse=True)
        
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances[:k]):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / math.log2(i + 1)
        return idcg
    
    def calculate_ndcg_at_k(self, retrieved_docs, relevant_docs, k):
        """Calculate Normalized DCG (NDCG) at rank k"""
        dcg = self.calculate_dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg = self.calculate_ideal_dcg_at_k(relevant_docs, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def evaluate_single_query(self, query_id, top_k=100):
        """Evaluate a single query and return metrics"""
        if query_id not in self.queries:
            print(f"❌ Query {query_id} not found")
            return None
        
        if query_id not in self.relevance_judgments:
            print(f"❌ No relevance judgments for query {query_id}")
            return None
        
        query_text = self.queries[query_id]
        relevant_docs = self.relevance_judgments[query_id]
        
        # Get BM25 results
        retrieved_docs = self.bm25_model.search(query_text, top_k=top_k)
        
        # Calculate metrics
        ap = self.calculate_average_precision(retrieved_docs, relevant_docs)
        rr = self.calculate_reciprocal_rank(retrieved_docs, relevant_docs)
        
        # Calculate NDCG at different cutoffs
        ndcg_1 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 1)
        ndcg_3 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 3)
        ndcg_5 = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, 5)
        
        # Calculate Precision and Recall at different cutoffs
        p_5 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5)
        p_10 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 10)
        p_20 = self.calculate_precision_at_k(retrieved_docs, relevant_docs, 20)
        
        r_5 = self.calculate_recall_at_k(retrieved_docs, relevant_docs, 5)
        r_10 = self.calculate_recall_at_k(retrieved_docs, relevant_docs, 10)
        r_20 = self.calculate_recall_at_k(retrieved_docs, relevant_docs, 20)
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'num_relevant': len(relevant_docs),
            'ap': ap,
            'rr': rr,
            'ndcg@1': ndcg_1,
            'ndcg@3': ndcg_3,
            'ndcg@5': ndcg_5,
            'p@5': p_5,
            'p@10': p_10,
            'p@20': p_20,
            'r@5': r_5,
            'r@10': r_10,
            'r@20': r_20,
            'retrieved_docs': retrieved_docs[:10]  # Top 10 for display
        }
    
    def evaluate_all_queries(self, top_k=100):
        """Evaluate all queries and return aggregated metrics"""
        print("🔄 Evaluating all queries...")
        
        all_results = []
        ap_scores = []
        rr_scores = []
        ndcg_1_scores = []
        ndcg_3_scores = []
        ndcg_5_scores = []
        
        for query_id in sorted(self.queries.keys()):
            if query_id in self.relevance_judgments:
                result = self.evaluate_single_query(query_id, top_k)
                if result:
                    all_results.append(result)
                    ap_scores.append(result['ap'])
                    rr_scores.append(result['rr'])
                    ndcg_1_scores.append(result['ndcg@1'])
                    ndcg_3_scores.append(result['ndcg@3'])
                    ndcg_5_scores.append(result['ndcg@5'])
        
        # Calculate aggregated metrics
        map_score = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
        mrr_score = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
        mean_ndcg_1 = sum(ndcg_1_scores) / len(ndcg_1_scores) if ndcg_1_scores else 0.0
        mean_ndcg_3 = sum(ndcg_3_scores) / len(ndcg_3_scores) if ndcg_3_scores else 0.0
        mean_ndcg_5 = sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0
        
        aggregated_results = {
            'num_queries_evaluated': len(all_results),
            'MAP': map_score,
            'MRR': mrr_score,
            'Mean_NDCG@1': mean_ndcg_1,
            'Mean_NDCG@3': mean_ndcg_3,
            'Mean_NDCG@5': mean_ndcg_5,
            'individual_results': all_results
        }
        
        return aggregated_results
    
    def print_evaluation_summary(self, results):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Number of queries evaluated: {results['num_queries_evaluated']}")
        print(f"MAP (Mean Average Precision): {results['MAP']:.4f}")
        print(f"MRR (Mean Reciprocal Rank): {results['MRR']:.4f}")
        print(f"Mean NDCG@1: {results['Mean_NDCG@1']:.4f}")
        print(f"Mean NDCG@3: {results['Mean_NDCG@3']:.4f}")
        print(f"Mean NDCG@5: {results['Mean_NDCG@5']:.4f}")
        print("="*60)
    
    def print_detailed_results(self, results, show_top_n=5):
        """Print detailed results for top N queries"""
        print(f"\n📋 DETAILED RESULTS (Top {show_top_n} queries by AP)")
        print("-"*80)
        
        # Sort by AP score
        sorted_results = sorted(results['individual_results'], 
                              key=lambda x: x['ap'], reverse=True)
        
        for i, result in enumerate(sorted_results[:show_top_n]):
            print(f"\n🔍 Query {result['query_id']}: {result['query_text'][:60]}...")
            print(f"   Relevant docs: {result['num_relevant']}")
            print(f"   AP: {result['ap']:.4f} | RR: {result['rr']:.4f}")
            print(f"   NDCG@1: {result['ndcg@1']:.4f} | NDCG@3: {result['ndcg@3']:.4f} | NDCG@5: {result['ndcg@5']:.4f}")
            print(f"   P@5: {result['p@5']:.4f} | P@10: {result['p@10']:.4f}")
            print(f"   R@5: {result['r@5']:.4f} | R@10: {result['r@10']:.4f}")
            
            print("   Top 5 retrieved docs:")
            for j, (doc_id, score) in enumerate(result['retrieved_docs'][:5]):
                print(f"     {j+1}. Doc {doc_id}: {score:.4f}")


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = IREvaluator()
    
    # Evaluate all queries
    results = evaluator.evaluate_all_queries(top_k=100)
    
    # Print results
    evaluator.print_evaluation_summary(results)
    evaluator.print_detailed_results(results, show_top_n=3)
    
    # Save results to file
    output_file = "results/evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
