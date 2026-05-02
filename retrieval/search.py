import scann

def build_index(embeddings):
    return scann.scann_ops_pybind.builder(
        embeddings, 10, "dot_product"
    ).build()

def search(index, query_vector):
    neighbors, _ = index.search(query_vector)
    return neighbors