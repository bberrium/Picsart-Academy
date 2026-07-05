---

## Clustering — Every Possible Exam Question

---

- [[#SECTION 1: What is Clustering & Why|SECTION 1: What is Clustering & Why]]
- [[#SECTION 2: Hard vs Soft / Hierarchical vs Partitional|SECTION 2: Hard vs Soft / Hierarchical vs Partitional]]
- [[#SECTION 3: Hierarchical Clustering|SECTION 3: Hierarchical Clustering]]
- [[#SECTION 4: K-Means|SECTION 4: K-Means]]
- [[#SECTION 5: K-Medoids|SECTION 5: K-Medoids]]
- [[#SECTION 6: Soft K-Means|SECTION 6: Soft K-Means]]
- [[#SECTION 7: Clustering Evaluation|SECTION 7: Clustering Evaluation]]

### SECTION 1: What is Clustering & Why

**Q1.** What is clustering and what makes a "good" cluster? State the two fundamental rules.

**Answer:** Clustering groups unlabeled data objects such that similar objects end up in the same group and dissimilar objects in different groups. The two rules: (1) objects within the same cluster should be close to each other (high cohesion); (2) objects in different clusters should be far apart (high separation). There is no single correct answer — the best grouping depends on the goal and the distance measure chosen.

---

**Q2.** Give four real-world applications of clustering from different domains.

**Answer:** (1) **Anomaly detection** — identifying fraudulent transactions or unusual network traffic by finding points that don't belong to any cluster. (2) **Market segmentation** — grouping customers by buying behavior to create targeted campaigns. (3) **Biology/medicine** — grouping patients with similar symptoms or identifying genes associated with diseases. (4) **Image processing** — grouping pixels with similar properties to identify objects, or finding dominant colors for recoloring.

---

**Q3.** Why is clustering considered subjective? Give a concrete example.

**Answer:** The "correct" clustering depends entirely on the goal, not the data alone. Example: grouping students for projects could be done by age, work experience, education level, or preferences — all are valid, but produce completely different groupings. The choice of features and distance measure encodes assumptions about what "similarity" means, which is a human decision.

---

### SECTION 2: Hard vs Soft / Hierarchical vs Partitional

**Q4.** Distinguish between hard and soft clustering. Give an example of each.

**Answer:** **Hard (non-fuzzy) clustering**: each object belongs to exactly one cluster — binary membership. Example: K-Means assigns each point to exactly one centroid. **Soft (fuzzy) clustering**: each object belongs to every cluster with some probability/weight (weight can be zero). Example: Soft K-Means assigns each point a "responsibility" value for each cluster, summing to 1.

---

**Q5.** Distinguish between partitional and hierarchical clustering.

**Answer:** **Partitional clustering**: finds a fixed number of flat clusters (e.g. K-Means produces K non-overlapping groups with no relationship between them). **Hierarchical clustering**: produces a nested sequence of clusterings — a tree where each level represents a different granularity of grouping. You can cut the tree at any level to get any number of clusters without rerunning the algorithm.

---

### SECTION 3: Hierarchical Clustering

**Q6.** Explain Agglomerative (bottom-up) hierarchical clustering step by step.

**Answer:** (1) Start: each data point is its own singleton cluster — N clusters. (2) Compute distances between all pairs of clusters. (3) Merge the two most similar (closest) clusters into one. (4) Recompute distances between the new cluster and all remaining clusters. (5) Repeat steps 3–4 until all points are in a single cluster. Result is a dendrogram where the height of each merge indicates the distance at which it occurred.

---

**Q7.** Explain Divisive (top-down) hierarchical clustering. What is one specific algorithm for it?

**Answer:** Start with all N points in one cluster. At each step, split one existing cluster into two daughter clusters — choosing the split that maximizes between-group dissimilarity. The MacNaughton-Smith algorithm: find the observation with the largest average dissimilarity from all others in the cluster; move it to a new cluster H; iteratively transfer observations from G to H if they are on average closer to H than to G; stop when no such observation remains. Repeat on each resulting cluster.

---

**Q8.** Describe the three linkage criteria for hierarchical clustering. What are the failure modes of each?

**Answer:**

- **Complete (max)** linkage: distance between clusters = max distance between any pair of points (one from each cluster). Failure: _chaining_ — clusters can be very spread out and non-compact; distant outliers can force premature merging.
- **Single (min)** linkage: distance = min distance between any pair of points. Failure: _crowding_ — clusters are compact but not well-separated; a point may be closer to members of another cluster than to some of its own.
- **Average** linkage: distance = average of all pairwise distances between the two clusters. Balanced approach — tends to produce clusters that are both compact and well-separated. Generally preferred.

---

**Q9.** List the pros and cons of hierarchical clustering.

**Answer:** **Pros:** (1) No need to predefine K — choose number of clusters after inspecting the dendrogram. (2) Dendrogram provides interpretable visualization. (3) Works well with histograms. (4) Hierarchy itself is useful in some problems (e.g. taxonomies). (5) Can use any distance measure. **Cons:** (1) O(n²) time complexity — cannot handle large datasets. (2) Hard to define exact number of clusters; sensitive to noise and outliers. (3) Poor cluster descriptors — hard to interpret what each cluster "means." (4) No corrections possible after merging/splitting — decisions are final.

---

### SECTION 4: K-Means

**Q10.** Describe the K-Means algorithm step by step. Why must data be standardized first?

**Answer:** (1) Standardize the data. (2) Choose K initial cluster centers (randomly or from data). (3) **Assignment step**: assign each point to the nearest centroid using Euclidean distance. (4) **Update step**: recompute each centroid as the mean of all points assigned to it. (5) Repeat 3–4 until assignments don't change (convergence). Standardization is required because K-Means uses Euclidean distance — features with large ranges dominate the distance calculation and effectively make other features irrelevant (same reason as KNN scaling).

---

**Q11.** Prove that K-Means is guaranteed to converge. Does it converge to a global minimum?

**Answer:** K-Means minimizes the within-cluster sum of squared distances J. Each step can only decrease J: the assignment step moves each point to the nearest centroid (any other assignment would increase J); the update step moves the centroid to the mean of its points (the mean minimizes squared distances, so any other center would increase J). Since J is bounded below by 0 and strictly decreases at each step (or stays the same), and there are finitely many possible assignments (K^N), the algorithm must terminate. **However**, J is non-convex — the algorithm converges to a local minimum, not necessarily the global minimum. Different initializations can yield different solutions.

---

**Q12.** What is the Elbow Method? Why can't you use raw inertia (WCSS) to compare models with different K?

**Answer:** Inertia always decreases as K increases — at K=N, every point is its own cluster and inertia=0. This is mathematically perfect but practically useless. You can't compare across K values because they solve different problems. The **Elbow Method** plots WCSS against K and looks for the inflection point (the "elbow") where increasing K gives diminishing returns in inertia reduction. The optimal K is at this bend. Limitation: in practice the elbow is often vague or absent — the curve decreases smoothly, making K selection subjective.

---

**Q13.** Why is K-Means sensitive to local minima, and how does K-Means++ address this?

**Answer:** K-Means is sensitive to initial centroid placement — bad initialization can trap the algorithm in a poor local minimum. **K-Means++** initialization: (1) Choose the first center uniformly at random from the data. (2) For each remaining point x, compute D(x) = distance to the nearest already-chosen center. (3) Choose the next center with probability proportional to D(x)² — points far from existing centers are more likely to be chosen. (4) Repeat until K centers are chosen. (5) Run standard K-Means from these centers. This spreads initial centroids far apart, dramatically reducing the probability of poor local minima and improving both speed and solution quality.

---

**Q14.** List the pros and cons of K-Means.

**Answer:** **Pros:** Simple and interpretable; fast and efficient; guaranteed convergence; scalable to large datasets; works well on globular, well-separated clusters. **Cons:** Requires predefined K; sensitive to initial centroid placement (local minima); assumes spherical (globular) clusters — fails on elongated or irregular shapes; doesn't work on categorical data (mean is undefined); sensitive to outliers (squared distances amplify their effect).

---

### SECTION 5: K-Medoids

**Q15.** What is K-Medoids and why was it introduced? How does it differ from K-Means?

**Answer:** K-Means uses the mean of assigned points as the cluster center — this requires squared Euclidean distance and is highly sensitive to outliers (squared distances give extreme points disproportionate influence). **K-Medoids** restricts cluster centers to be actual data points (medoids) — the observation in each cluster that minimizes the total dissimilarity to all other points in the cluster. Differences: (1) centers are real data points, not computed means; (2) any dissimilarity measure can be used, not just squared Euclidean; (3) more robust to outliers — one extreme point can't pull the center far from the data. Tradeoff: computationally more expensive (explicit optimization over all candidate points per cluster).

---

**Q16.** Describe the K-Medoids algorithm step by step.

**Answer:** (1) Choose K, the number of clusters. (2) Initialize K medoids randomly from the data points. (3) **Assignment step**: assign each point to the nearest medoid according to the chosen dissimilarity measure. (4) **Update step**: for each cluster, find the point within the cluster that minimizes the total dissimilarity to all other points in that cluster — this becomes the new medoid. (5) Repeat steps 3–4 until medoids don't change (convergence).

---

### SECTION 6: Soft K-Means

**Q17.** Explain Soft K-Means. How does the assignment step differ from hard K-Means?

**Answer:** Instead of assigning each point to exactly one cluster (hard assignment), Soft K-Means assigns each point n a "responsibility" r_{nk} for each cluster k — a soft degree of membership between 0 and 1, summing to 1 across clusters. The responsibility is computed as: r_{nk} ∝ exp(−β · ||x_n − m_k||²), where β controls the "softness" — higher β makes assignments harder (closer to K-Means), lower β makes them softer (more uniform). The update step moves each centroid to the weighted mean of all points, weighted by their responsibilities. Con: β must be set as a hyperparameter; also can't handle clusters with unequal weight or width well.

---

### SECTION 7: Clustering Evaluation

**Q18.** What is the difference between internal and external clustering evaluation?

**Answer:** **Internal evaluation**: no ground-truth labels available. Quality is assessed using only the data's own structure — distances, densities, variance (e.g. Inertia, Silhouette, Calinski-Harabasz, Davies-Bouldin). **External evaluation**: ground-truth labels exist (e.g. from a survey or expert annotation). Quality is measured by how well cluster assignments match the true labels (e.g. ARI, NMI). External evaluation is more trustworthy but labels are rarely available in unsupervised settings.

---

**Q19.** Define the Silhouette Coefficient. Write the formula and interpret s(i)≈1, s(i)≈0, s(i)<0.

**Answer:** For each point i: let a(i) = mean distance to all other points in the same cluster (intra-cluster cohesion); b(i) = mean distance to all points in the nearest other cluster (inter-cluster separation).

**s(i) = (b(i) − a(i)) / max(a(i), b(i))**

- s(i) ≈ 1: point is tightly packed in its own cluster and far from all others — well clustered.
- s(i) ≈ 0: point sits exactly on the boundary between two clusters — ambiguous assignment.
- s(i) < 0: point is on average closer to another cluster than its own — likely misassigned.

The overall Silhouette score = mean s(i) across all points. Range: [−1, 1]; higher is better.

---

**Q20.** Explain the Calinski-Harabasz (CH) Index. When would you use it over Silhouette?

**Answer:** CH = (between-cluster dispersion) / (within-cluster dispersion), scaled by cluster/sample counts. It measures how much the clusters are separated relative to how compact they are internally — the same logic as trace(B)/trace(W) from Gaussian Discriminant Analysis. **Higher CH = better defined clusters.** Use CH over Silhouette when: the dataset is massive — CH is computationally much faster than Silhouette (which requires all pairwise distances). Limitation: like all internal metrics, CH assumes clusters are roughly convex and may favor spherical, well-separated clusters.

---

**Q21.** Explain the Davies-Bouldin Index. How does its interpretation differ from Silhouette and CH?

**Answer:** For each cluster Ci, find the most similar neighboring cluster Cj — measured by the ratio (si + sj)/dij where si, sj are the internal scatters (average within-cluster distances) and dij is the distance between centroids. DB = average of these worst-case ratios across all clusters. **Unlike Silhouette and CH where higher is better, a lower DB index is better** — it means the worst-case cluster overlap is minimized. DB is useful because it explicitly focuses on the hardest cases (the most confused pair of clusters), giving a conservative, pessimistic quality estimate.

---

**Q22.** Why can't standard Accuracy be used to evaluate clustering even when ground-truth labels exist? What metric is used instead?

**Answer:** Cluster IDs are arbitrary — "Cluster 0" doesn't inherently correspond to "Class A." A perfect clustering could assign Class A → Cluster 2, Class B → Cluster 0, and Accuracy would be 0% despite being correct. The **Adjusted Rand Index (ARI)** solves this: it looks at all pairs of points and asks — for each pair, does the clustering agree with the true labels on whether they belong together or apart? It's permutation-invariant (label assignment doesn't matter). The "adjusted" version subtracts the expected RI under random chance, so ARI≈0 means random, ARI=1 means perfect, ARI<0 means worse than random.

---

**Q23.** Explain Normalized Mutual Information (NMI). What concept from Decision Trees is it connected to?

**Answer:** NMI measures how much information the cluster assignments C provide about the true labels Y, normalized to [0,1]. NMI = I(Y;C) / √(H(Y)·H(C)), where I(Y;C) is mutual information and H is entropy — the same entropy formula used in Decision Tree splitting. NMI=0 means cluster assignments tell you nothing about the true labels; NMI=1 means knowing the cluster assignment perfectly determines the true label. Unlike ARI which counts pairs, NMI uses an information-theoretic lens — it measures reduction in uncertainty.

---

**Q24.** Compare K-Means and Hierarchical Clustering directly across 5 dimensions.

**Answer:**

||K-Means|Hierarchical|
|---|---|---|
|**K required upfront?**|Yes|No — choose after dendrogram|
|**Complexity**|O(NKI) — fast|O(N²) — slow for large data|
|**Cluster shape**|Spherical only|Any shape (linkage-dependent)|
|**Result type**|Flat partition|Nested tree (dendrogram)|
|**Corrections**|Can reinitialize|Merges/splits are final|

---

**Q25.** A dataset has K-Means running with K=3 but the same three clusters always appear regardless of initialization. What does this suggest about the data?

**Answer:** This suggests the data has three well-separated, compact, roughly spherical clusters — the global minimum of the K-Means objective is easily found from any starting point because the clusters are strongly defined. When K-Means is robust to initialization, it is a sign that the true cluster structure aligns well with K-Means' assumptions (Euclidean distance, spherical clusters, similar sizes). If initialization mattered a lot, it would suggest overlapping, irregular, or unequally-sized clusters where K-Means gets stuck in different local minima depending on where centroids start.