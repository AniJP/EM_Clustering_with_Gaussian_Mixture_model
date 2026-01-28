# EM Clustering on Wikipedia Dataset (Data 606 Final Assignment)

This project implements the Expectation-Maximization (EM) algorithm from scratch for clustering Wikipedia articles represented as high-dimensional TF-IDF vectors. The clustering is modeled using a Gaussian Mixture Model (GMM) with **diagonal covariance matrices** for efficiency and stability.

---

## 📁 Repository Structure

```
.
├── em_wiki_indexing.ipynb       # Main notebook for running the pipeline
├── em_algorithm.py               # Main EM implementation
├── index_generator.py            # Index-to-word mapping utility
├── output_formatter.py           # Formats final outputs
├── visualizer.py                 # Generates ASCII word clouds
├── em_parameters.txt             # Learned GMM parameters
├── cluster_assignments.txt       # Final document-to-cluster assignments
├── cluster_stats.txt             # Summary stats per cluster
├── convergence_log.txt           # Log-likelihood values per iteration
├── ascii_wordclouds.txt          # Cluster-specific top-word visualizations
├── 4_map_index_to_word.json      # Index-to-word mapping used for interpretation
├── benchmark.py                  # Extra credit: benchmark with sklearn GMM
└── index_to_word.json.zip        # (Optional) Compressed mapping file
```

---

## Project Goals

- Implement EM for GMM clustering (from scratch)
- Use diagonal covariance matrices for scalable high-dimensional modeling
- Extract and interpret top cluster words using TF-IDF features
- Visualize clusters using ASCII word clouds
- Compare against `sklearn` GMM for benchmarking (extra credit)

---

## Methodology

### 1. **Data Preparation**
- Articles were preprocessed and converted into TF-IDF vectors using `sklearn.TfidfVectorizer`.
- All feature vectors were normalized to unit length.
- Vocabulary indices were stored and mapped using `4_map_index_to_word.json`.

### 2. **EM Initialization**
- Cluster means were initialized via K-Means.
- Cluster weights were initialized uniformly.
- Diagonal covariance matrices were computed with small variance floors (e.g., 1e-8).

### 3. **EM Iterative Optimization**
- **E-step**: Responsibilities (`r_ik`) computed using Gaussian likelihoods.
- **M-step**: Means, variances (diagonal only), and cluster weights updated.
- The algorithm ran for 20 iterations or until convergence (log-likelihood stabilization).

### 4. **Postprocessing and Output**
- Each document assigned to its most probable cluster.
- Top 5 and 10 words (by mean value) extracted per cluster.
- ASCII word clouds generated to show word prominence.

---

## Results

### Convergence
- Log-likelihood increased monotonically from `2.29e9` to `2.35e9`, showing stable convergence.
- Convergence log is available in `convergence_log.txt`.

### Cluster Parameters
- All cluster weights are within `[0.11, 0.26]` and sum to ~1.0.
- Top words per cluster include consistent, interpretable terms.
- Cluster stats and top words are reported in:
  - `em_parameters.txt`
  - `cluster_stats.txt`

### Visualization
- ASCII word clouds (`ascii_wordclouds.txt`) highlight dominant terms per cluster, scaled by mean frequency.
- Example:  
```
Cluster 0:
antiaids antiaids antiaids teambefore montrauxjoby airgate airgate
```

### Assignments
- `cluster_assignments.txt` lists each document and its assigned cluster.
- Formatted per assignment requirement.

---

## Benchmarking

- The script `benchmark.py` compares this EM implementation with `sklearn.mixture.GaussianMixture`.
- Metrics compared:
  - Runtime
  - Memory usage
  - Final log-likelihood

---

## How to Run the Project

1. Open the notebook:
   ```
   em_wiki_indexing.ipynb
   ```

2. Follow the cells sequentially to:
   - Load and preprocess TF-IDF data
   - Initialize and run your custom EM algorithm (`em_algorithm.py`)
   - Format and output cluster assignments/statistics (`output_formatter.py`)
   - Visualize results as ASCII word clouds (`visualizer.py`)
   - Optionally, compare results with sklearn GMM (`benchmark.py`)

3. All results are saved to:
   - `cluster_assignments.txt`
   - `cluster_stats.txt`
   - `em_parameters.txt`
   - `ascii_wordclouds.txt`
   - `convergence_log.txt`

Make sure the JSON mapping file `4_map_index_to_word.json` is in the working directory for word label interpretation.

---

## Authors

- Student Names: *Anirudh Jayaprakash* and *Pramod Kumar*
- University IDs: *120622342* *121032911*
- Course: Data 606 – Algorithms for Data Science
- Submission Date: May 17, 2025

---

## 📬 Contact

For any questions regarding this project, please reach out to:
`pramod21@umd.edu` or `ani31399@umd.edu`

---

##  Sample Output Demonstration 

To help understand how the EM algorithm works, we included a small-scale demonstration using a **subset of the Wikipedia dataset** (`people_wiki.csv`). This run produces the same types of output as the full dataset but in a faster, more interpretable way.

Files generated include:
- `cluster_assignments.txt`
- `cluster_stats.txt`
- `em_parameters.txt`
- `ascii_wordclouds.txt`
- `convergence_log.txt`

This sample output serves as a useful reference for verifying output structure, inspecting convergence behavior, and interpreting cluster themes on a manageable dataset.
