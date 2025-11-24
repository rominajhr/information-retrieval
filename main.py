import re
import json
import random
import string
import heapq
from collections import defaultdict, Counter
from typing import List, Any, Optional

# ------------------------- Tokenizer / Normalizer -------------------------

class SimpleTokenizer:
    def __init__(self, keep_punctuation: bool = False):
        self.keep_punctuation = keep_punctuation
        # basic regex to remove punctuation (keeps unicode letters/digits/spaces)
        self.punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

    def normalize(self, text: str) -> str:
        text = text.lower()
        if not self.keep_punctuation:
            text = self.punct_re.sub(' ', text)
        # collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        norm = self.normalize(text)
        if norm == '':
            return []
        return norm.split(' ')

# ------------------------- Simple B-Tree Implementation -------------------------

class BTreeNode:
    def __init__(self, t: int, leaf: bool = False):
        self.t = t  # minimum degree
        self.leaf = leaf
        self.keys: List[str] = []  # list of keys (terms)
        self.values: List[Any] = []  # associated values with keys
        self.children: List['BTreeNode'] = []  # child node references

    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1

    def __repr__(self):
        if self.leaf:
            return f"Leaf(keys={self.keys})"
        return f"Node(keys={self.keys}, leaf={self.leaf})"

class BTree:
    def __init__(self, t: int = 2):
        if t < 2:
            raise ValueError('B-Tree minimum degree t must be >= 2')
        self.t = t
        self.root: BTreeNode = BTreeNode(t, leaf=True)

    def search(self, key: str) -> Optional[Any]:
        return self._search(self.root, key)

    def _search(self, node: BTreeNode, key: str) -> Optional[Any]:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        if node.leaf:
            return None
        return self._search(node.children[i], key)

    def insert(self, key: str, value: Any):
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_nonfull(new_root, key, value)
        else:
            self._insert_nonfull(root, key, value)

    def _split_child(self, parent: BTreeNode, i: int):
        t = self.t
        node_to_split = parent.children[i]
        new_node = BTreeNode(t, leaf=node_to_split.leaf)

        # new_node gets t-1 keys from node_to_split
        new_node.keys = node_to_split.keys[t:]
        new_node.values = node_to_split.values[t:]

        # if not leaf, transfer children
        if not node_to_split.leaf:
            new_node.children = node_to_split.children[t:]

        # shrink node_to_split
        mid_key = node_to_split.keys[t-1]
        mid_val = node_to_split.values[t-1]
        node_to_split.keys = node_to_split.keys[:t-1]
        node_to_split.values = node_to_split.values[:t-1]
        if not node_to_split.leaf:
            node_to_split.children = node_to_split.children[:t]

        # insert new child into parent
        parent.children.insert(i+1, new_node)
        parent.keys.insert(i, mid_key)
        parent.values.insert(i, mid_val)

    def _insert_nonfull(self, node: BTreeNode, key: str, value: Any):
        i = len(node.keys) - 1
        if node.leaf:
            # find position
            while i >= 0 and key < node.keys[i]:
                i -= 1
            if i >= 0 and key == node.keys[i]:
                # replace value
                node.values[i] = value
                return
            node.keys.insert(i+1, key)
            node.values.insert(i+1, value)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full():
                self._split_child(node, i)
                # after split, the middle key moves up and may affect i
                if key > node.keys[i]:
                    i += 1
            self._insert_nonfull(node.children[i], key, value)

    def traverse(self) -> List[tuple]:
        """Return list of (key, value) in sorted order."""
        res: List[tuple] = []
        self._traverse_node(self.root, res)
        return res

    def _traverse_node(self, node: BTreeNode, res: List[tuple]):
        for i, k in enumerate(node.keys):
            if not node.leaf:
                self._traverse_node(node.children[i], res)
            res.append((k, node.values[i]))
        if not node.leaf:
            self._traverse_node(node.children[len(node.keys)], res)

    def pretty_print(self):
        lines = []
        self._pretty_print_node(self.root, lines, level=0)
        return '\n'.join(lines)

    # === ASCII GRAPHICAL PRINT ===
    def ascii_print(self):
        """Return an ASCII-art style representation of the B-Tree."""
        lines = []
        self._ascii_node(self.root, prefix="", is_tail=True, out=lines)
        return "\n".join(lines)

    def _ascii_node(self, node: BTreeNode, prefix: str, is_tail: bool, out: List[str]):
        """Helper (recursive) to build ASCII lines. Kept as an instance method so
        it can access node attributes via 'node' and keep consistent indentation."""
        out.append(prefix + ("└── " if is_tail else "├── ") + str(node.keys))
        if not node.leaf:
            for i, child in enumerate(node.children):
                last = (i == len(node.children) - 1)
                self._ascii_node(child, prefix + ("    " if is_tail else "│   "), last, out)

    def _pretty_print_node(self, node: BTreeNode, lines: List[str], level: int):
        indent = '  ' * level
        lines.append(f"{indent}Node(level={level}, keys={node.keys})")
        if not node.leaf:
            for child in node.children:
                self._pretty_print_node(child, lines, level+1)
    
    def ascii_collect(self):
        lines = []
        self._ascii_collect_node(self.root, 0, lines)
        return lines

    def _ascii_collect_node(self, node, level, lines):
        indent = "  " * level
        lines.append(f"{indent}{node.keys}")
        if not node.leaf:
            for child in node.children:
                self._ascii_collect_node(child, level + 1, lines)

# ------------------------- Inverted Index -------------------------

class InvertedIndex:
    def __init__(self, btree_degree: int = 2):
        # inverted index in-memory representation: term -> set(docIDs)
        self.index = defaultdict(set)
        self.tokenizer = SimpleTokenizer()
        # B-Tree dictionary for term -> posting_list
        self.btree = BTree(t=btree_degree)
        # stats
        self.doc_count = 0
        self.term_frequencies = Counter()

    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenizer.tokenize(text)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[token] += tokens.count(token)
        self.doc_count += 1

    def finalize(self):
        # convert posting sets to sorted lists and insert to B-Tree
        for term, docset in self.index.items():
            posting = sorted(list(docset))
            # store posting lists as list of docIDs
            self.btree.insert(term, posting)

    def search(self, term: str) -> Optional[List[str]]:
        term_norm = self.tokenizer.normalize(term)
        return self.btree.search(term_norm)

    def save(self, path: str):
        # we'll save the traverse list as a dict: term -> posting list
        data = {k: v for k, v in self.btree.traverse()}
        meta = {
            'doc_count': self.doc_count,
            'terms_count': len(data)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'meta': meta, 'dictionary': data}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.doc_count = obj['meta'].get('doc_count', 0)
        dictionary = obj['dictionary']
        # rebuild tree
        self.btree = BTree(t=self.btree.t)
        for term in sorted(dictionary.keys()):
            self.btree.insert(term, dictionary[term])

    def top_k_terms(self, k: int = 10) -> List[tuple]:
        # return k terms with highest frequency using heap
        if k <= 0:
            return []
        return heapq.nlargest(k, self.term_frequencies.items(), key=lambda x: x[1])

    def print_index(self, limit: int = 50):
        print(f"Inverted Index (showing up to {limit} terms):")
        for i, (term, posting) in enumerate(self.btree.traverse()):
            if i >= limit:
                break
            print(f"{term} -> {posting}")

# ------------------------- Document generation / I/O helpers -------------------------

LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
).split()

# generate a random document with approx n_tokens tokens

BASE_WORDS = [
    "system","model","random","vector","matrix","token","language","semantic","python","data",
    "index","tree","algorithm","search","compute","neural","layer","network","memory","buffer",
    "encode","decode","parser","query","process","signal","feature","object","storage","design",
    "pattern","entropy","value","kernel","archive","lookup","mapping","embedding","gradient"
]

import string

# Expand vocabulary to thousands of unique words
EXTENDED_VOCAB = set(BASE_WORDS)

# shuffled variants
for w in BASE_WORDS:
    for _ in range(20):
        EXTENDED_VOCAB.add(''.join(random.sample(w, len(w))))

# prefix/suffix synthesizing
for w in BASE_WORDS:
    for _ in range(20):
        p = ''.join(random.choices(string.ascii_lowercase, k=3))
        s = ''.join(random.choices(string.ascii_lowercase, k=3))
        EXTENDED_VOCAB.add(p + w + s)

EXTENDED_VOCAB = list(EXTENDED_VOCAB)


def generate_random_document(n_tokens: int = 3000, vocabulary: Optional[List[str]] = None) -> str:
    """Generates a document with 3000 tokens using thousands of unique words."""
    if vocabulary is None:
        vocabulary = EXTENDED_VOCAB

    tokens = [random.choice(vocabulary) for _ in range(n_tokens)]

    # sprinkle punctuation to test tokenizer
    for i in range(0, len(tokens), 150):
        tokens[i] += random.choice([",", ".", ";"])

    return " ".join(tokens)

# load plain text file

def load_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ------------------------- Demo / Main -------------------------

def demo_build_and_show(num_documents: int = 3, tokens_each: int = 3000, btree_t: int = 3):
    print('--- Building standard inverted index (no stemming, no positional info) ---')
    ii = InvertedIndex(btree_degree=btree_t)

    docs = {}
    for i in range(1, num_documents+1):
        doc_id = f'doc{i}'
        txt = generate_random_document(n_tokens=tokens_each)
        docs[doc_id] = txt
        ii.add_document(doc_id, txt)
        print(f'Added {doc_id} ({tokens_each} tokens approx)')

    ii.finalize()

    

    return ii, docs

def save_full_output(INVERTED_INDEX, DOCUMENTS, all_terms, filename="full_output.txt"):
    with open(filename, "w", encoding="utf-8", newline="\n") as f:

        # --- DOCUMENTS ---
        f.write("=== GENERATED DOCUMENTS ===\n\n")
        for name, text in DOCUMENTS.items():
            f.write(f"--- {name} ---\n")
            f.write(text) 
            f.write(f"\n\n(total {len(text.split())} tokens)\n\n")
            f.write("=" * 80 + "\n\n")

        # --- TERMS ---
        f.write("\n=== ALL TERMS (SORTED) ===\n")
        for t in all_terms:
            f.write(t + "\n")

        f.write("\n" + "="*80 + "\n\n")

        # --- POSTING LISTS ---
        f.write("=== FULL POSTING LISTS ===\n\n")
        for t in all_terms:
            posting = INVERTED_INDEX.search(t)
            f.write(f"{t}: {posting}\n")

        f.write("\n" + "="*80 + "\n\n")

        # --- ASCII B-TREE ---
        f.write("=== ASCII B-TREE ===\n\n")
        lines = INVERTED_INDEX.btree.ascii_collect()
        for line in lines:
            f.write(line + "\n")

    print(f"🔵 Full output saved to: {filename}")
    
if __name__ == '__main__':
    # Build the inverted index and documents
    INVERTED_INDEX, DOCUMENTS = demo_build_and_show(
        num_documents=3,
        tokens_each=3000,
        btree_t=3
    )


    print("\n===================== FULL OUTPUT =====================\n")

    # === All generated documents ===
    print("=== GENERATED DOCUMENTS (FULL CONTENT) ===")
    for name, text in DOCUMENTS.items():
        print(f"\n--- {name} ---")
        print(text)
        print(f"(total {len(text.split())} tokens)")

    # === All terms extracted in sorted order ===
    print("\n=== ALL TERMS (SORTED) ===")
    all_terms = [term for term, _ in INVERTED_INDEX.btree.traverse()]
    for t in all_terms:
        print(t)

    # === Full inverted index: EVERY term -> posting list ===
    print("\n=== FULL INVERTED INDEX (term → posting list) ===")
    for term, posting in INVERTED_INDEX.btree.traverse():
        print(f"{term} -> {posting}")

    # === B-tree traversal (sorted keys) ===
    print("\n=== B-TREE TRAVERSE (sorted key order) ===")
    traversal = INVERTED_INDEX.btree.traverse()
    for k, v in traversal:
        print(f"{k}: {v}")


    # === ASCII graphical B-tree ===
    print("\n=== B-TREE ASCII ART ===")
    print(INVERTED_INDEX.btree.ascii_print())
    save_full_output(INVERTED_INDEX, DOCUMENTS, all_terms)


    print("\n===================== END FULL OUTPUT =====================")
