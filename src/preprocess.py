import re, os, string, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn","get","got","also","even","us","would","could","still","back",
    "well","much","way","made","said","like","make","see","go","going","come",
    "came","know","think","one","time","new","good","first","last","long",
    "great","little","own","right","big","high","old","any","day","man","work",
    "use","used","using","want","need","may","might","put","take","give","keep",
    "let","seems","seem","something","anything","everything","nothing",
    "someone","anyone","everyone","lot","really","already","always",
    "never","ever","often","usually","today","yesterday","tomorrow","via",
}

_SUFFIXES = ("ing","ed","ly","es","s","er","est","ion","tion","ness")

def simple_stem(word):
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word)-len(suf)>=3:
            return word[:-len(suf)]
    return word

def clean_text(text):
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"@\w+","",text)
    text = re.sub(r"#(\w+)",r"\1",text)
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t)>2]
    tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)

def full_preprocess(text):
    return tokenize_and_lemmatize(clean_text(text))

def build_tfidf(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def run_preprocessing(
    raw_path=os.path.join("data","social_media_posts.csv"),
    out_path=os.path.join("data","processed_posts.csv"),
    vec_path=os.path.join("models","tfidf_vectorizer.pkl"),
):
    print("📂 Loading raw dataset …")
    df = pd.read_csv(raw_path)
    print(f"   Rows loaded: {len(df)}")
    print("🧹 Cleaning text …")
    df["cleaned_text"] = df["text"].apply(clean_text)
    print("🔤 Preprocessing (tokenize + stem) …")
    df["processed_text"] = df["cleaned_text"].apply(tokenize_and_lemmatize)
    print("📊 Building TF-IDF features …")
    vectorizer, X = build_tfidf(df["processed_text"])
    print(f"   Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"   Feature matrix  : {X.shape}")
    os.makedirs("data",exist_ok=True)
    os.makedirs("models",exist_ok=True)
    df.to_csv(out_path,index=False)
    with open(vec_path,"wb") as f: pickle.dump(vectorizer,f)
    print(f"\n✅ Processed CSV  → {out_path}")
    print(f"✅ TF-IDF model   → {vec_path}")
    cols=["text","cleaned_text","processed_text","sentiment"]
    print(df[cols].head(3).to_string(index=False))
    return df, vectorizer, X

if __name__=="__main__":
    run_preprocessing()