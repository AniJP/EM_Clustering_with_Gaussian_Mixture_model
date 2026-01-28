import numpy as np

def save_ascii_wordclouds(means, index_to_word, filename="ascii_wordclouds.txt", top_n=15):
    with open(filename, "w") as f:
        for k in range(len(means)):
            f.write(f"Cluster {k} ASCII Word Cloud:\n")
            top_indices = np.argsort(means[k])[::-1][:top_n]

            for idx in top_indices:
                word = index_to_word.get(idx, f"[word_{idx}]")
                count = int(means[k][idx] * 1000)
                ascii_word = word * max(1, min(count, 20))
                f.write(f"{ascii_word}\n")

            f.write("\n" + "=" * 40 + "\n\n")

    print(f"ASCII word clouds saved to {filename}")
