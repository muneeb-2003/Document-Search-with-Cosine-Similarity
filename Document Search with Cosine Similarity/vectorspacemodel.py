import os
from tkinter import *
from tkinter import scrolledtext, messagebox
from nltk.stem import PorterStemmer
import re
from collections import Counter
import math

# Function to load stop words from a file
def load_stop_words(stop_words_file):
    with open(stop_words_file, "r") as f:
        return set(word.strip() for word in f.readlines())

# Function to preprocess text
def preprocess_text(text, stop_words):
    ps = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize and convert to lowercase
    tokens = [ps.stem(token) for token in tokens if token not in stop_words]  # Remove stop words and stem
    return tokens

# Function to build inverted index
def build_inverted_index(folder_path, stop_words_file):
    stop_words = load_stop_words(stop_words_file)
    inverted_index = {}
    document_word_counts = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = preprocess_text(text, stop_words)

                doc_id = filename.split('.')[0]  # Extract document ID from filename
                document_word_counts[doc_id] = Counter(tokens)  # Count word occurrences for each document

                for token in set(tokens):
                    if token not in inverted_index:
                        inverted_index[token] = {doc_id}
                    else:
                        inverted_index[token].add(doc_id)

    return inverted_index, document_word_counts

# Function to save index to file
def save_index_to_file(inverted_index, document_word_counts, file_path):
    with open(file_path, 'w') as f:
        f.write("Inverted Index:\n")
        for term, postings in inverted_index.items():
            f.write(f"{term}: {', '.join(postings)}\n")

        f.write("\nDocument Word Counts:\n")
        for doc_id, counts in document_word_counts.items():
            f.write(f"{doc_id}: {dict(counts)}\n")

# Function to load index from file
def load_index_from_file(file_path):
    inverted_index = {}
    document_word_counts = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        index_type = None
        for line in lines:
            line = line.strip()
            if line == "Inverted Index:":
                index_type = "inverted"
                continue
            elif line == "Document Word Counts:":
                index_type = "document"
                continue
            
            if index_type == "inverted":
                term, postings = line.split(': ')
                inverted_index[term] = set(postings.split(', '))
            elif index_type == "document":
                doc_id, counts = line.split(': ')
                document_word_counts[doc_id] = Counter(eval(counts))

    return inverted_index, document_word_counts

# Function to calculate TF-IDF
def calculate_tf_idf(inverted_index, document_word_counts, total_documents):
    tf_idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log(total_documents / (df + 1))  # Adding 1 to avoid division by zero
        for doc_id in postings:
            tf = document_word_counts[doc_id][term] / sum(document_word_counts[doc_id].values())
            if doc_id not in tf_idf:
                tf_idf[doc_id] = {}
            tf_idf[doc_id][term] = tf * idf

    return tf_idf

# Function to calculate cosine similarity
def cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector[term] * document_vector.get(term, 0) for term in query_vector)
    query_norm = sum(value ** 2 for value in query_vector.values()) ** 0.5
    document_norm = sum(value ** 2 for value in document_vector.values()) ** 0.5
    if query_norm == 0 or document_norm == 0:
        return 0
    else:
        return dot_product / (query_norm * document_norm)

# Function to process the query
def process_query(query, inverted_index, document_vectors, alpha, top_n=5):
    ps = PorterStemmer()
    query_tokens = preprocess_text(query, stop_words)
    query_vector = Counter(ps.stem(token) for token in query_tokens)  # Create query vector

    scores = {}
    for term in query_vector:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                if doc_id not in scores:
                    scores[doc_id] = cosine_similarity(query_vector, document_vectors[doc_id])
                else:
                    scores[doc_id] += cosine_similarity(query_vector, document_vectors[doc_id])

    # Filter results based on alpha threshold
    filtered_results = {doc_id: score for doc_id, score in scores.items() if score >= alpha}

    if not filtered_results:  # Check if there are no matching documents
        messagebox.showinfo("No Results", "No matching documents found.")
        return []

    # Sort results based on similarity scores
    sorted_results = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)

    # Select top-n documents as the champions list
    champions_list = sorted_results[:top_n]

    return champions_list

# Main function
def main():
    global stop_words
    folder_path = "ResearchPapers"  
    stop_words_file = 'Stopword-List.txt'  
    index_file_path = 'index.txt'  

    inverted_index, document_word_counts = build_inverted_index(folder_path, stop_words_file)
    total_documents = len(document_word_counts)
    tf_idf = calculate_tf_idf(inverted_index, document_word_counts, total_documents)

    document_vectors = {doc_id: Counter(tf_idf[doc_id]) for doc_id in tf_idf}

    alpha = 0.05  # Alpha threshold for filtering search results

    # Save the indexes to a file
    save_index_to_file(inverted_index, document_word_counts, index_file_path)
    print("Indexes saved successfully.")

    # Create Tkinter GUI
    root = Tk()
    root.title("Document Search")
    root.configure(bg='#F0F0F0')  # Set background color

    # Search bar
    query_label = Label(root, text="Enter your query:", bg='#F0F0F0', font=("Arial", 12))
    query_label.grid(row=0, column=0, padx=5, pady=5)
    query_entry = Entry(root, width=50, font=("Arial", 12))
    query_entry.grid(row=0, column=1, padx=5, pady=5)

    # Top N Selector
    top_n_label = Label(root, text="Top N Results:", bg='#F0F0F0', font=("Arial", 12))
    top_n_label.grid(row=0, column=2, padx=5, pady=5)
    top_n_entry = Entry(root, width=10, font=("Arial", 12))
    top_n_entry.grid(row=0, column=3, padx=5, pady=5)
    top_n_entry.insert(END, "5")  # Default value

    # Filter Threshold
    alpha_label = Label(root, text="Filter Threshold:", bg='#F0F0F0', font=("Arial", 12))
    alpha_label.grid(row=0, column=4, padx=5, pady=5)
    alpha_entry = Entry(root, width=10, font=("Arial", 12))
    alpha_entry.grid(row=0, column=5, padx=5, pady=5)
    alpha_entry.insert(END, "0.05")  # Default value

    # Results display
    results_text = scrolledtext.ScrolledText(root, width=80, height=20, font=("Arial", 12))
    results_text.grid(row=1, column=0, columnspan=6, padx=10, pady=5)

    # Function to search documents
    def search_documents():
        query = query_entry.get()
        top_n = int(top_n_entry.get())
        alpha = float(alpha_entry.get())
        results_text.delete(1.0, END)  # Clear previous results
        results = process_query(query, inverted_index, document_vectors, alpha, top_n)
        if results:
            results_text.insert(END, "Search Results:\n")
            for doc_id, score in results:
                results_text.insert(END, f"Document ID: {doc_id}, Score: {score}\n")
        else:
            results_text.insert(END, "No matching documents found.")

    # Search button
    search_button = Button(root, text="Search", command=search_documents, bg='#4CAF50', fg='white', font=("Arial", 12))
    search_button.grid(row=0, column=6, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    stop_words = load_stop_words('Stopword-List.txt')
    main()
