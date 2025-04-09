import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Function to read the article from a file
def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()  # Read the entire file content
        article = filedata.split(". ")  # Split by sentence delimiter
        sentences = []
        for sentence in article:
            sentence = sentence.replace("[^a-zA-Z]", " ").strip()  # Remove non-alphabet characters
            if sentence:  # Ensure sentence is not empty
                sentences.append(sentence.split(" "))
        return sentences

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Function to generate the similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # Avoid self-comparison
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Function to generate the summary
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Read the article
    sentences = read_article(file_name)
    print(f"Sentences extracted: {sentences}")  # Debug print

    # Handle empty input
    if not sentences:
        print("Error: No sentences found in the input file.")
        return

    # Generate similarity matrix
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    print(f"Similarity Matrix: {sentence_similarity_matrix}")  # Debug print

    # Create a graph and apply PageRank
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print(f"Ranked Sentences: {ranked_sentences}")  # Debug print

    # Handle top_n exceeding sentence count
    top_n = min(top_n, len(ranked_sentences))

    # Pick top N sentences
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    # Print summary
    print("Summary:", ". ".join(summarize_text))

# Add the file name and run the summary generation
if __name__ == "__main__":
    file_name = "article.txt"  # Ensure this file exists in the same directory
    generate_summary(file_name, top_n=5)
