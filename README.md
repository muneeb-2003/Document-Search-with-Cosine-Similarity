# Document Search with Cosine Similarity "VectorSpace Model"

This project implements a document search engine that ranks documents based on their relevance to a given query. The ranking is performed using **Cosine Similarity** combined with **TF-IDF (Term Frequency-Inverse Document Frequency)** weighting to compute the relevance between documents and the query. The system includes a simple **GUI (Graphical User Interface)** built using **Tkinter**, allowing users to interact with the document search engine in an easy-to-use manner.

## Features

- **Document Search**: Search a collection of `.txt` documents based on a query.
- **Cosine Similarity**: Measures the similarity between the query and each document.
- **TF-IDF**: Computes the weight of each term in a document based on its frequency and rarity.
- **Stop Word Removal**: Common stop words like "the", "and", "is", etc., are filtered out to enhance search results.
- **Stemming**: Words are reduced to their base or root form using **Porter Stemmer**.
- **GUI**: A simple Tkinter-based GUI to input queries and display search results.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- **Tkinter** (usually comes with Python by default)
- **NLTK** (Natural Language Toolkit)
- **os** (for file operations)
- **math** (for mathematical operations)
- **collections** (for counting word frequencies)

To install **NLTK**, use the following command:

```bash
pip install nltk
