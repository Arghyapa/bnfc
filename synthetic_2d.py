# Package Import
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, MeanShift, estimate_bandwidth, DBSCAN, OPTICS, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score, rand_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import networkx as nx
from sklearn.metrics import pairwise_distances
from itertools import combinations

#!pip install ripser

#!pip install gudhi

'''
"""src/main/resources/datasets/artificial/disk-6000n.arff"""

# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np
data = np.loadtxt('disk_6k.txt')
true_labels = np.loadtxt('disk6k_label.txt')

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()

"""src/main/resources/datasets/artificial/3MC.arff"""

# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np
data = np.loadtxt('3mc1.txt')
true_labels = np.loadtxt('3mc_label.txt')



data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()

"""src/main/resources/datasets/artificial/smile1.arff"""

# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np
data = np.loadtxt('smile1.txt')
true_labels = np.loadtxt('smile_label.txt')


data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()


"""src/main/resources/datasets/artificial/2d-20c-no0.arff"""
# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np
data = np.loadtxt('2d20cn0.txt')
true_labels = np.loadtxt('2d20c_label.txt')


data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='tab20')
plt.show()


"""src/main/resources/datasets/artificial/donutcurves.arff"""

# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np
data = np.loadtxt('donutcurve1.txt')
true_labels = np.loadtxt('donut_label.txt')

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()


# # Generate synthetic data with n clusters from clustering benchmark
import numpy as np
data = np.loadtxt('cure-4k1.txt')
true_labels = np.loadtxt('cure4k_label.txt')

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()

'''
"""src/main/resources/datasets/artificial/complex9.arff"""

# # Generate synthetic data with 3 clusters from clustering benchmark
import numpy as np


data = np.loadtxt('complex.txt')
true_labels = np.loadtxt('complex_label.txt')

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = true_labels
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='tab10')
plt.show()


from ripser import ripser

"""##BNFC"""

num_clust = int(input("Number of Clusters:"))

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def extract_betti(PD, scale_seq):
    b = np.zeros(len(scale_seq))
    for k in range(len(scale_seq)):
        b[k] = np.sum((scale_seq[k] >= PD[:, 0]) & (scale_seq[k] < PD[:, 1]))
    return b

def weighted_knn(data, nKNN):
    knn = NearestNeighbors(n_neighbors=nKNN).fit(data)
    dists, ind = knn.kneighbors(data)

    # Compute weights inversely proportional to distances
    weights = 1 / (dists + 1e-5)  # Adding a small value to avoid division by zero

    return ind, weights

def bnfc_weighted(data, nKNN, filt_len=100):
    N = data.shape[0]

    # Perform weighted k-NN
    ind, weights = weighted_knn(data, nKNN)  # Get indices and weights from weighted knn
    maxscale = np.max(1 / weights[:, -1])  # Max of inverse of weights (distances)

    scale_seq = np.linspace(0, maxscale, filt_len)

    # Initialize arrays to store Betti numbers
    betti_0 = np.zeros((N, filt_len))
    betti_1 = np.zeros((N, filt_len))

    for i in range(N):

        neighborhood = data[ind[i,]]

        # Compute persistent homology
        rips = ripser(neighborhood, maxdim=1)
        PD_0 = rips['dgms'][0]
        PD_1 = rips['dgms'][1]

        # Replace infinite persistence values with maxscale for Betti-0 & Betti-1
        PD_0[PD_0[:, 1] == np.inf, 1] = maxscale
        PD_1[PD_1[:, 1] == np.inf, 1] = maxscale

        # Extract Betti-0 and Betti-1 numbers for this point over the scale sequence
        betti_0[i, :] = extract_betti(PD_0, scale_seq)
        betti_1[i, :] = extract_betti(PD_1, scale_seq)

    cos_scores_0= np.zeros((N, nKNN))
    cos_scores_1= np.zeros((N, nKNN))

    # Compute cosine similarity between data points based on their Betti numbers
    for i in range(N):
        # Get Betti numbers for the current point and its neighbors
        current_betti_0 = betti_0[i, :].reshape(1, -1)
        current_betti_1 = betti_1[i, :].reshape(1, -1)

        neighbors_betti_0 = betti_0[ind[i,], :]
        neighbors_betti_1 = betti_1[ind[i,], :]

        # Compute cosine similarity for Betti-0, Betti-1, and Betti-2 arrays
        sim_0 = cosine_similarity(current_betti_0, neighbors_betti_0).flatten()
        cos_scores_0[i,:]= sim_0

        sim_1 = cosine_similarity(current_betti_1, neighbors_betti_1).flatten()
        cos_scores_1[i,:]= sim_1

    # Initialize adjacency matrix
    A = np.zeros((N, N))

    # Calculate thresholds using the IQR method (Interquartile Range)
    tao_1 = np.percentile(cos_scores_0, 75) + 1.5 * (np.percentile(cos_scores_0, 75) - np.percentile(cos_scores_0, 25))
    tao_0 = np.percentile(cos_scores_0, 25) - 1.5 * (np.percentile(cos_scores_0, 75) - np.percentile(cos_scores_0, 25))
    meu_1 = np.percentile(cos_scores_1, 75) + 1.5 * (np.percentile(cos_scores_1, 75) - np.percentile(cos_scores_1, 25))
    meu_0 = np.percentile(cos_scores_1, 25) - 1.5 * (np.percentile(cos_scores_1, 75) - np.percentile(cos_scores_1, 25))

    for i in range(N):
        indices_1 = ind[i, cos_scores_0[i,] >= tao_0]
        indices_2 = ind[i, cos_scores_1[i,] >= meu_0]


        # Find the common neighbors that meet the threshold in all three Betti dimensions
        common_indices = np.intersect1d(indices_1, indices_2)

        # Form adjacency matrix by connecting similar points
        A[i, common_indices] = 1

    return A


from numpy.linalg import eig
rng = np.random.default_rng()

def tpspect(data, nKNN):

  def calculate_U(X, V):
      n, d = X.shape
      U = np.zeros(n)
      for i in range(n):
          dist = ((X[i] - V) * (X[i] - V)).sum(1)
          j = dist.argmin()
          U[i] =j
      return U

  def calculate_V(X, U, k):
      n, d = X.shape
      V = np.zeros([k, d])
      for j in range(k):
          index = np.where(U == j)[0]
          r = index.shape[0]
          if r == 0:
              V[j] = rng.choice(X)
          else:
              for i in index:
                  V[j] = V[j] + X[i]
              V[j] = V[j] / r
      return V

  def k_means(X, k):
      n, d = X.shape
      U = np.zeros(n)
      V = rng.choice(X, k)
      condition = True
      while (condition):
          temp = np.copy(U)
          U = calculate_U(X, V)
          V = calculate_V(X, U, k)
          condition = not np.array_equal(U, temp)
      return U

  k = num_clust
  n = data.shape[0]

  A = bnfc_weighted(data, nKNN)

  Adj = np.zeros((n, n))
  sigma = np.std(data)
  for i in range(n):
    for j in range(n):
      if A[i, j]==1:
        Adj[i, j] = np.exp(-np.linalg.norm(data[i] - data[j])**2 / (2*(sigma**2)))*A[i,j]*A[j,i]

  D = np.diag(np.sum(Adj, axis=1))

  L = D - Adj
  eigenvalues, eigenvectors = eig(L)

  indices = np.argsort(eigenvalues)[:k]
  U = eigenvectors[:,indices]

  bnfc_label = k_means(U,k)

  return bnfc_label

"""##BNFC"""

# List to store ARI scores
ari_scores = []

# Range of nKNN values to test
nKNN_values = range(1, 25)  # Adjust the range as needed

# Perform clustering and compute ARI for each nKNN
for nKNN in nKNN_values:
    Tplabel = tpspect(data, nKNN)
    ari = adjusted_rand_score(true_labels, Tplabel)
    ari_scores.append(ari)

# Plot the ARI scores
plt.figure(figsize=(8, 6))
plt.plot(nKNN_values, ari_scores, marker='o')
plt.xlabel('nKNN')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('ARI for Different nKNN Values')
plt.grid(True)
plt.show()

nKNN = np.argmax(ari_scores)+1
#nKNN = int(input("KNN:"))
bnfc_label = tpspect(data, nKNN)

data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = bnfc_label
plt.scatter(data_df['x'], data_df['y'],c= data_df['labels'], cmap='viridis')
plt.show()

bnfc_ari = adjusted_rand_score(true_labels, bnfc_label)
bnfc_ri = rand_score(true_labels, bnfc_label)
bnfc_nmi = normalized_mutual_info_score(true_labels, bnfc_label)

print(f'BNFC RI: {bnfc_ri:.3f}, BNFC ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')



'''
# Define K-means clustering
def kmeans_clustering(data, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

"""##Minibatchkmeans"""

# Define MiniBatchkmeans clustering
def minibatchkmeans(data, num_clusters=3):
    minibatchkmeans= MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', max_iter=20, random_state=42)
    minibatchkmeans.fit(data)
    return minibatchkmeans.labels_

"""##Affinity Propagation"""

# define Affinity Propagation
def affinity_propagation(data):
    affinity = AffinityPropagation(damping= 0.6, max_iter= 200, convergence_iter= 50, affinity= 'precomputed')
    affinity.fit(data)
    return affinity.labels_

"""##Optics"""

## define Optics Clustering
def optics(data):
    optics= OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)
    optics.fit(data)
    return optics.labels_

"""##Agglomerative Clustering"""

## define Agglomerative Clustering
def agglomerative_clustering(data, num_clusters=20):
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='single')
    agglomerative.fit(data)
    return agglomerative.labels_

"""##Spectral"""

# Define Spectral clustering
def spectral_clustering(data, num_clusters=3):
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
    spectral.fit(data)
    return spectral.labels_

"""##Tomato"""

# using ToMATo method
from gudhi.clustering.tomato import Tomato

# define ToMATo clustering
def Tomato_clustering(data):
    t = Tomato()
    t.fit(data)
    t.n_clusters_= 20
    return t.labels_


"""##Mean-shift"""

# Define Mean Shift Clustering
def mean_shift(data):
    mean_shift = MeanShift()
    mean_shift.fit(data)
    return mean_shift.labels_

"""##DBSCAN"""

# Define DBSCAN clustering
def dbscan_clustering(data, eps=0.5, min_samples=4):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data_scaled)
    return dbscan.labels_

"""#Run Different clustering

##Tomato
"""

# Run Tomato clustering
t_labels = Tomato_clustering(data)

"""##Optics"""

# Run Optics
optics_labels = optics(data)

"""##Agglomerative"""

# Run Agglomerative Clustering
agglomerative_labels = agglomerative_clustering(data,num_clusters=20)

"""##Minibatchkmeans"""

# Run Minibatchkmeans
minibatchkmeans_labels = minibatchkmeans(data, num_clusters=20)

# # run Affinity Propagation
# affinity_labels = affinity_propagation(Data)

"""##K-means"""

# Run K-means clustering
kmeans_labels = kmeans_clustering(data, num_clusters=20)

"""##Spectral"""

# Run spectral clustering
spectral_labels = spectral_clustering(data, num_clusters=20)

"""##Mean-shift"""

# Run Mean-Shift Clustering
mean_shift_labels = mean_shift(data)

"""##DBSCAN"""

# Run DBSCAN Clustering
dbscan_labels = dbscan_clustering(data, eps=0.5, min_samples=20)

"""#Visualisation of different clusters"""

# Plotting results for visual comparison
fig, ax = plt.subplots(1,7, figsize=(12,4))

# Ground-Truth
data_df['labels'] = true_labels
ax[0].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='tab20')
ax[0].set_title('Ground Truth')

# # K-means
# data_df = pd.DataFrame(data, columns=['x', 'y'])
# data_df['labels'] = kmeans_labels
# ax[1].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
# ax[1].set_title('K-means')

# Spectral
data_df['labels'] = spectral_labels
ax[2].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[2].set_title('Spectral')

# Mean-Shift
data_df['labels'] = mean_shift_labels
ax[3].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[3].set_title('Mean-Shift')

# DBSCAN
data_df['labels'] = dbscan_labels
ax[4].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[4].set_title('DBSCAN')

# Tomato
data_df['labels'] = t_labels
ax[5].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[5].set_title('ToMATo')

# MiniBatchKmeans
data_df['labels'] = minibatchkmeans_labels
ax[1].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[1].set_title('MiniBatchKmeans')

# BNFC
data_df['labels'] = bnfc_label
ax[6].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[6].set_title('BNFC')

plt.tight_layout()
plt.show()

# Plotting results for visual comparison
fig, ax = plt.subplots(1,9, figsize=(18,4))

# Ground-Truth
data_df['labels'] = true_labels
ax[0].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[0].set_title('Ground Truth')

# K-means
data_df = pd.DataFrame(data, columns=['x', 'y'])
data_df['labels'] = kmeans_labels
ax[1].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[1].set_title('K-means')

# Spectral
data_df['labels'] = spectral_labels
ax[2].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[2].set_title('Spectral')

# Mean-Shift
data_df['labels'] = mean_shift_labels
ax[3].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[3].set_title('Mean-Shift')

# DBSCAN
data_df['labels'] = dbscan_labels
ax[4].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[4].set_title('DBSCAN')

# Optics
data_df['labels'] = optics_labels
ax[5].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[5].set_title('Optics')

# Agglomerative
data_df['labels'] = agglomerative_labels
ax[6].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[6].set_title('Agglomerative')

# Tomato
data_df['labels'] = t_labels
ax[7].scatter(data_df['x'], data_df['y'], c= data_df['labels'], cmap='viridis')
ax[7].set_title('ToMATo')

# # MiniBatchKmeans
# data_df['labels'] = minibatchkmeans_labels
# ax[8].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
# ax[8].set_title('MiniBatchKmeans')

# BNFC
data_df['labels'] = bnfc_label
ax[8].scatter(data_df['x'], data_df['y'], c=data_df['labels'], cmap='viridis')
ax[8].set_title('BNFC')

plt.tight_layout()
plt.show()

kmeans_ri = adjusted_rand_score(true_labels, kmeans_labels)
kmeans_ari = rand_score(true_labels, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_labels)

spectral_ari = adjusted_rand_score(true_labels, spectral_labels)
spectral_ri = rand_score(true_labels, spectral_labels)
spectral_nmi = normalized_mutual_info_score(true_labels, spectral_labels)

bnfc_ari = adjusted_rand_score(true_labels, bnfc_label)
bnfc_ri = rand_score(true_labels, bnfc_label)
bnfc_nmi = normalized_mutual_info_score(true_labels, bnfc_label)

mean_shift_ari = adjusted_rand_score(true_labels, mean_shift_labels)
mean_shift_ri = rand_score(true_labels, mean_shift_labels)
mean_shift_nmi = normalized_mutual_info_score(true_labels, mean_shift_labels)

dbscan_ari = adjusted_rand_score(true_labels, dbscan_labels)
dbscan_ri = rand_score(true_labels, dbscan_labels)
dbscan_nmi = normalized_mutual_info_score(true_labels, dbscan_labels)

optics_ari = adjusted_rand_score(true_labels, optics_labels)
optics_ri = rand_score(true_labels, optics_labels)
optics_nmi = normalized_mutual_info_score(true_labels, optics_labels)

ac_ari =adjusted_rand_score(true_labels, agglomerative_labels)
ac_ri = rand_score(true_labels, agglomerative_labels)
ac_nmi = normalized_mutual_info_score(true_labels, agglomerative_labels)

tomato_ari = adjusted_rand_score(true_labels, t_labels)
tomato_ri = rand_score(true_labels, t_labels)
tomato_nmi = normalized_mutual_info_score(true_labels, t_labels)

# ap_ari = adjusted_rand_score(true_labels, affinity_labels)
# ap_ri = rand_score(true_labels, affinity_labels)
# ap_nmi = normalized_mutual_info_sc ore(true_labels, affinity_labels)

minibatchkmeans_ari = adjusted_rand_score(true_labels, minibatchkmeans_labels)
minibatchkmeans_ri = rand_score(true_labels, minibatchkmeans_labels)
minibatchkmeans_nmi = normalized_mutual_info_score(true_labels, minibatchkmeans_labels)



print(f'Kmeans RI: {kmeans_ri:.3f}, Kmeans ARI: {kmeans_ari:.3f}, NMI: {kmeans_nmi: .3f}')
print(f'Spectral RI: {spectral_ri:.3f}, Spectral ARI: {spectral_ari:.3f}, NMI: {spectral_nmi: .3f}')
print(f'BNFC RI: {bnfc_ri:.3f}, BNFC ARI: {bnfc_ari:.3f}, NMI: {bnfc_nmi: .3f}')
print(f'ToMATo RI: {tomato_ri:.3f}, ToMATo ARI: {tomato_ari:.3f}, NMI: {tomato_nmi: .3f}')
print(f'Mean-Shift RI: {mean_shift_ri:.3f}, Mean-Shift ARI: {mean_shift_ari:.3f}, NMI: {mean_shift_nmi: .3f}')
print(f'DBSCAN RI: {dbscan_ri:.3f}, DBSCAN ARI: {dbscan_ari:.3f}, NMI: {dbscan_nmi: .3f}')
print(f'Optics RI: {optics_ri:.3f}, Optics ARI: {optics_ari:.3f}, NMI: {optics_nmi: .3f}')
print(f'Agglomerative RI: {ac_ri:.3f}, Agglomerative ARI: {ac_ari:.3f}, NMI: {ac_nmi: .3f}')
print(f'Minibatchkmeans RI: {minibatchkmeans_ri:.3f}, Minibatchkmeans ARI: {minibatchkmeans_ari:.3f}, Minibatchkmeans NMI: {minibatchkmeans_nmi: .3f}')
'''
