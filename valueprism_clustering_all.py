import os
import pickle

import pandas as pd
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from sentence_transformers import SentenceTransformer


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    "hf://datasets/allenai/ValuePrism/full/full.csv",
    dtype={"vrd": str, "text": str}
)

# All Values, Rights and Duties are clustered together
texts = df["text"].dropna().unique().tolist()

print(f"Total unique statements: {len(texts)}")


# ============================================================
# Hugging Face embedding model
# ============================================================

# MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # smaller model for testing

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True
)

print(f"Using embedding model: {MODEL_NAME}")


# ============================================================
# Encoding
# ============================================================

VECTOR_FILE = "vectors_nomic_v1.5.pkl"

if os.path.exists(VECTOR_FILE):
    vectors = pickle.load(open(VECTOR_FILE, "rb"))
else:
    vectors = {}


# Only encode statements that are not already cached
texts_to_encode = [
    text for text in texts
    if text not in vectors
]

print(f"Statements to encode: {len(texts_to_encode)}")


if texts_to_encode:

    # Nomic recommends the "clustering:" prefix when
    # embeddings are being used for clustering.
    clustering_texts = [
        f"clustering: {text}"
        for text in texts_to_encode
    ]

    embeddings = model.encode(
        clustering_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    for text, embedding in zip(texts_to_encode, embeddings):
        vectors[text] = embedding.tolist()


pickle.dump(vectors, open(VECTOR_FILE, "wb"))


# ============================================================
# Clustering
# ============================================================

THRESHOLD = 0.8

print("\nClustering all Values, Rights and Duties together...")

ids = list(vectors.keys())
embeddings = [vectors[text] for text in ids]

# Hierarchical clustering
#
# Since embeddings are normalized, cosine distance is more
# appropriate for semantic clustering than Ward distance.
Z = linkage(
    embeddings,
    method="average",
    metric="cosine"
)

labels = fcluster(
    Z,
    t=THRESHOLD,
    criterion="distance"
)

print(f"Number of clusters: {len(set(labels))}")


# ============================================================
# Create clusters
# ============================================================

clusters = {}

for text, cluster_label in zip(ids, labels):

    if cluster_label not in clusters:
        clusters[cluster_label] = []

    clusters[cluster_label].append(text)


# ============================================================
# Calculate clustroids
# ============================================================

print("Calculating clustroids...")

clustroids = {}

for cluster_label, items in tqdm(clusters.items()):

    clustroid = None
    min_distance = float("inf")

    for item in items:

        # Sum squared Euclidean distance to all other
        # items in the cluster.
        distance = sum(
            (
                vectors[item][j] - vectors[other][j]
            ) ** 2
            for other in items
            for j in range(len(vectors[item]))
        )

        if distance < min_distance:
            min_distance = distance
            clustroid = item

    clustroids[cluster_label] = clustroid


# ============================================================
# Cluster quality
# ============================================================

print("Calculating cluster quality...")

cluster_quality = {}

for label, items in tqdm(clusters.items()):

    centroid = clustroids[label]

    # --------------------------------------------------------
    # Intra-cluster distance
    # --------------------------------------------------------

    intra_distance = sum(
        (
            vectors[item][j] - vectors[centroid][j]
        ) ** 2
        for item in items
        for j in range(len(vectors[item]))
    )

    # --------------------------------------------------------
    # Inter-cluster distance
    # --------------------------------------------------------

    inter_distance = sum(
        (
            vectors[centroid][j]
            - vectors[clustroids[other]][j]
        ) ** 2
        for other in clusters
        if other != label
        for j in range(len(vectors[centroid]))
    )

    quality = inter_distance / (intra_distance + 1e-10)

    cluster_quality[label] = {
        "intra_distance": intra_distance,
        "inter_distance": inter_distance,
        "quality": quality
    }


# ============================================================
# Create cluster dataframe
# ============================================================

df_clusters = pd.DataFrame(
    [
        {
            "Cluster Label": label,
            "Clustroid": clustroids[label],
            "Intra-distance": cluster_quality[label]["intra_distance"],
            "Inter-distance": cluster_quality[label]["inter_distance"],
            "Quality": cluster_quality[label]["quality"],
            "Size": len(clusters[label])
        }
        for label in clusters
    ]
)

df_clusters.to_csv(
    f"clusters_nomic_t{THRESHOLD}.csv",
    index=False
)


# ============================================================
# Add cluster information to original dataframe
# ============================================================

text_to_cluster = {
    text: cluster_label
    for text, cluster_label in zip(ids, labels)
}

df["Cluster Label"] = df["text"].map(text_to_cluster)

df["Clustroid"] = df["Cluster Label"].map(
    clustroids
)

df.to_csv(
    f"valueprism_nomic_t{THRESHOLD}.csv",
    index=False
)


# ============================================================
# Print some cluster examples
# ============================================================

print("\nCluster examples:")

for label, items in list(clusters.items())[:10]:

    print(f"\nCluster {label} ({len(items)} items)")
    print(f"Clustroid: {clustroids[label]}")

    for item in items[:5]:
        print(f"  - {item}")


print("\nDone.")
print(
    f"Clusters saved to: "
    f"clusters_nomic_t{THRESHOLD}.csv"
)
print(
    f"Dataset saved to: "
    f"valueprism_nomic_t{THRESHOLD}.csv"
)