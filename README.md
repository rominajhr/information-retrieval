# Inverted Index with B-Tree (Python)

This project implements a **simple Information Retrieval system** in Python. It builds an **Inverted Index** over a collection of text documents and stores the dictionary (term → posting list) inside a **B-Tree** data structure.

The project is educational and suitable for:

* Information Retrieval courses
* Data Structures (B-Tree)
* Search & Indexing demonstrations

---

## ✨ Features

* Text normalization and tokenization
* In-memory inverted index (term → document IDs)
* Custom implementation of a **B-Tree**
* Efficient term lookup using the B-Tree
* ASCII visualization of the B-Tree structure
* Saving full output (documents, terms, postings, tree) to a file

---

## 📁 Project Structure

```
project/
│
├── main.py            # Main source code
├── README.md          # This file
└── docs/              # Input documents
    ├── doc1.txt
    ├── doc2.txt
    └── doc3.txt
```

Each `.txt` file inside the `docs` directory is treated as **one document**.
The filename (without extension) is used as the document ID.

---

## 🧠 Core Components

### 1. SimpleTokenizer

* Converts text to lowercase
* Removes punctuation
* Splits text into tokens based on whitespace

```python
SimpleTokenizer(keep_punctuation=False)
```

---

### 2. B-Tree Implementation

A custom B-Tree with configurable minimum degree `t`.

* Supports insertion and search
* Stores:

  * `key` → term (string)
  * `value` → posting list (list of document IDs)

Key methods:

* `insert(key, value)`
* `search(key)`
* `traverse()`
* `ascii_print()`

---

### 3. InvertedIndex

Builds and manages the inverted index.

Responsibilities:

* Tokenizing documents
* Building term → document mappings
* Inserting terms into the B-Tree
* Searching for terms

Key methods:

```python
add_document(doc_id, text)
finalize()
search(term)
save(path)
load(path)
```

---

## 🚀 How to Run

### 1. Prepare Input Documents

Create a directory named `docs` and add text files:

```
docs/
├── doc1.txt
├── doc2.txt
└── doc3.txt
```

Example content:

```
Information retrieval systems rely on indexing and search.
```

---

### 2. Run the Program

```bash
python main.py
```

---

## 📤 Output

The program produces:

1. Printed output in terminal:

   * All documents
   * All extracted terms (sorted)
   * Full inverted index
   * B-Tree traversal
   * ASCII B-Tree visualization

2. A file named:

```
full_output.txt
```

This file contains:

* Full document contents
* All terms
* Posting lists
* ASCII representation of the B-Tree

---

## 🌳 Example ASCII B-Tree Output

```
['algorithm', 'data']
  ['buffer', 'compute']
  ['index', 'memory', 'search']
```

(This structure depends on input data and B-Tree degree.)

---

## ⚙️ Configuration

You can adjust the B-Tree minimum degree:

```python
INVERTED_INDEX, DOCUMENTS = demo_build_and_show_from_files(
    input_dir="docs",
    btree_t=3
)
```

* Larger `t` → wider, shallower tree
* Smaller `t` → narrower, deeper tree

---

## 📚 Educational Notes

* This implementation does **not** include:

  * Stemming
  * Stop-word removal
  * Positional indexes
  * TF-IDF weighting

These can be added as future extensions.

---

## 🛠 Possible Extensions

* Positional inverted index
* TF / DF / TF-IDF scoring
* Query processing (AND / OR)
* Disk-based B-Tree
* Support for Persian text

---

## 👩‍💻 Author

Developed as an educational project for learning:

* Inverted Indexes
* B-Trees
* Information Retrieval fundamentals

---

## ❤️ Final Note

This project prioritizes **clarity and learning** over performance.
Perfect for understanding how real search engines build indexes internally.
