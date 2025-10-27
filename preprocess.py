import os
import re
import json
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Load Cranfield dataset
def load_cranfield_data(path="./dataset/cranfield/docs"):
    docs = []
    for file in sorted(os.listdir(path), key=lambda x: int(re.sub(r'\D', '', x))):  # sắp xếp số tăng dần
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8", errors="ignore") as f:
                docs.append({
                    "doc_id": re.sub(r'\D', '', file),
                    "text": f.read()
                })
    print(f"Loaded {len(docs)} documents from Cranfield dataset")
    return docs

# Normalize
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
# Tokenization
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# Remove Stopwords + POS Filtering
def remove_stopwords(tokens, stop_words):
    tokens_pos = pos_tag(tokens)
    filtered = [
        w for w, pos in tokens_pos
        if pos.startswith(('N', 'J', 'V')) and w.isalpha() and w not in stop_words
    ]
    return filtered

# Chuyển đổi POS tag của NLTK sang POS của WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return NOUN

# Lemmatization
def lemmatize(tokens, lemmatizer):
    tokens_pos = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tokens_pos
    ]
    return lemmatized

# Preprocess
def preprocess(text, stop_words, lemmatizer):
    text = normalize(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stop_words)
    tokens = lemmatize(tokens, lemmatizer)
    return tokens

# Sinh bigram toàn corpus
def build_global_bigrams(all_tokens, min_freq=5, top_n=500):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)
    finder.apply_freq_filter(min_freq)
    scored = finder.score_ngrams(bigram_measures.pmi)
    top_bigrams = {' '.join(b): True for b, _ in scored[:top_n]}
    print(f"Generated {len(top_bigrams)} bigrams.")
    return top_bigrams

# Tiền xử lý toàn corpus
def preprocess_corpus(docs):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    all_tokens = []
    for d in docs:
        tokens = preprocess(d["text"], stop_words, lemmatizer)
        all_tokens.extend(tokens)
    
    # Xây bigram toàn corpus
    global_bigrams = build_global_bigrams(all_tokens)

    processed_docs = []
    for d in tqdm(docs, desc="Processing Cranfield docs"):
        tokens = preprocess(d["text"], stop_words, lemmatizer)
        # Sinh bigram trong tài liệu nếu khớp với global bigram
        doc_bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in global_bigrams:
                doc_bigrams.append(bigram)
        all_terms = tokens + doc_bigrams

        processed_docs.append({
            "doc_id": int(d["doc_id"]),
            "terms": all_terms
        })
    processed_docs.sort(key=lambda x: x["doc_id"])
    return processed_docs

# Lưu lại
def save_preprocessed_data(processed_docs, out_path="./dataset/preprocessed_cranfield.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)
    print(f"Saved preprocessed dataset → {out_path}")

# Run
if __name__ == "__main__":
    cranfield_docs = load_cranfield_data("./dataset/cranfield/docs")
    processed = preprocess_corpus(cranfield_docs)
    save_preprocessed_data(processed)
    print(f"Example doc {processed[0]['doc_id']}: {processed[0]['terms'][:30]}")
