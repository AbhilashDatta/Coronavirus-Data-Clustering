# 19CS30001: Abhilash Datta
# DC1 [Mini Project 3]
# Coronavirus Data Clustering using Complete Linkage Hierarchical Clustering Technique

import pandas as pd
import numpy as np
import time
start_time = time.time()
np.random.seed(2)

#Z score normalization
def normalize(train_np):
  train_np = (train_np - np.mean(train_np,axis = 0))/np.std(train_np,axis = 0)

#part 1
class K_Means:

  def __init__(self, X, k = 3, iterations = 20):
    self.k = k
    self.n = X.shape[0]
    self.X = X
    self.iters = iterations

  def get_initial_centroids(self):
    lst = []
    count = 0
    for _ in range(self.k):
      i = np.random.randint(0,self.n)
      lst.append(self.X[i])

    return np.array(lst)     

  def get_distance(self,d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)
    return np.linalg.norm(d1-d2)

  def get_clusters(self, centroids):
    cluster = dict()
    for i in range(self.k):
      cluster[i] = []

    for d in self.X:
      min_d = float('inf')
      for i in range(len(centroids)):
        if self.get_distance(centroids[i],d) < min_d:
          min_d = self.get_distance(centroids[i],d)  
          c = i
      
      cluster[c].append(d)

    return cluster

  def get_clusters_with_datapoint_indices(self,centroids):
    cluster = dict()
    for i in range(self.k):
      cluster[i] = []

    for d in range(len(self.X)):
      min_d = float('inf')
      for i in range(len(centroids)):
        if self.get_distance(centroids[i],self.X[d]) < min_d:
          min_d = self.get_distance(centroids[i],self.X[d])  
          c = i

      cluster[c].append(d)

    return cluster

  def update_centroids(self, clusters, old_centroids):
    centroids = []
    for key in clusters.keys():
      if len(clusters[key]) != 0:
        centroids.append(np.mean(clusters[key],axis = 0))
      else :
        centroids.append(old_centroids[key])

    return centroids   

  def final_centroids(self):

    initial_centroids = self.get_initial_centroids()
    old_centroids = initial_centroids

    for i in range(self.iters):
      clusters = self.get_clusters(old_centroids)
      new_centroids = self.update_centroids(clusters,old_centroids)
      old_centroids = new_centroids

    return new_centroids

#Storing in a file
def store_kmeans_clusters(kmeans_clusters):
  fp = open('kmeans.txt','w')
  f = 0
  for i in kmeans_clusters[1]:
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.write('\n')
  f = 0
  for i in kmeans_clusters[2]:
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.write('\n')
  f = 0
  for i in kmeans_clusters[0]:
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.close()

#part 2
class Evaluate:

  def __init__(self, model):
    self.model = model
  
  def cohesion(self,cluster,i):
    d = 0
    for j in cluster:
      if i is not j:
        d += self.model.get_distance(i,j)
    if len(cluster)==1: 
      return 1 
      
    return d/(len(cluster)-1)
    
  def separation(self,cluster_id,i,clusters):
    min_b = float('inf')
    for k in clusters.keys():
      d = 0
      if k!= cluster_id:
        for j in clusters[k]:
          d += self.model.get_distance(i,j)
        b = d/len(clusters[k])
        if b<min_b:
          min_b = b

    return min_b
  
  def silhouette_coefficient(self):
    max_sc = -float('inf')
    final_centroids = self.model.final_centroids()
    final_clusters = self.model.get_clusters(final_centroids)

    for k in final_clusters.keys():
      cluster = final_clusters[k]
      s = 0
      for i in cluster:
        b = self.separation(k,i,final_clusters)
        a = self.cohesion(cluster,i)
        s += (b-a)/max(b,a)
      
      s_tilde = s/len(cluster)
      if s_tilde>max_sc:
        max_sc = s_tilde
    
    return max_sc

#part 3
def print_best_k():
  print("\tFinding Optimal Value of K\n")
  max_value = -float('inf')

  for K in range(3,7):
      k_means = K_Means(train_np,k = K)
      eval = Evaluate(k_means)
      sc = eval.silhouette_coefficient()
      if max_value < sc:
          max_value = sc
          best_k = K
      print("Silhouette Coefficient for k =",K,"is",sc)
      
  print("Optimal number of clusters =",best_k)

#part 4
class Agglomerative:

  def __init__(self, X, k = 3):
    self.k = k
    self.n = X.shape[0]
    self.X = X
  
  def get_euclidean_distance(self,d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)
    return np.linalg.norm(d1-d2)

  def get_distance(self,c1,c2):
    max_d = -float('inf')
    for i in c1:
      for j in c2:
        if self.get_euclidean_distance(i,j)>max_d:
          max_d = self.get_euclidean_distance(i,j)
    return max_d
  
  def combine_clusters(self,k1,k2,clusters,dist_matrix):
    c = []
    for i in clusters[k1]:
      c.append(i)
    for i in clusters[k2]:
      c.append(i)

    clusters.pop(k1)
    clusters.pop(k2)
    clusters[min(k1,k2)] = c

    dist_matrix.pop(k1)
    dist_matrix.pop(k2)
    dist_matrix[min(k1,k2)] = dict()

    #Complete linkage (max distance)
    for i in dist_matrix.keys():
      if i!= min(k1,k2):
        dist_matrix[min(k1,k2)][i] = max(dist_matrix[i][k1],dist_matrix[i][k2])
        dist_matrix[i][min(k1,k2)] = max(dist_matrix[i][k1],dist_matrix[i][k2])
        dist_matrix[i].pop(max(k1,k2))

  def find_closest(self, dist_matrix):
    min_d = float('inf')
    pi = 0
    pj = 0
    for i in dist_matrix.keys():
      for j in dist_matrix[i].keys():
        if dist_matrix[i][j]<min_d and j>i:
          min_d = dist_matrix[i][j]
          pi = i
          pj = j
    return (pi,pj)

  def get_clusters(self):
    clusters = dict()

    for i in range(len(self.X)):
      clusters[i] = [self.X[i]]

    dist_matrix = dict()
    for i in range(self.n):
      dist_matrix[i] = dict()
      for j in range(self.n):
        if i!=j:
          dist_matrix[i][j] = self.get_distance(clusters[i],clusters[j])
    
    count = self.n

    while count!=self.k :
      k1,k2 = self.find_closest(dist_matrix)
      self.combine_clusters(k1,k2,clusters,dist_matrix)
      count -= 1

    new = dict()
    i = 0
    for k in clusters.keys():
      new[i] = clusters[k]
      i += 1

    return new

#Storing agglomerative clusters
def store_agg_clusters(agg_clusters):
  fp = open('agglomerative.txt','w')
  f = 0
  for i in sorted(agg_clusters[0]):
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.write('\n')
  f = 0
  for i in sorted(agg_clusters[2]):
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.write('\n')
  f = 0
  for i in sorted(agg_clusters[1]):
    if f==0: 
      fp.write(str(i))
      f = 1
    else:
      fp.write(', '+str(i))

  fp.close()

#Jaccard Similarity
class Jaccard:

  def __init__(self, kmeans_clusters, agg_clusters):
    self.kmeans_clusters = kmeans_clusters
    self.agg_clusters = agg_clusters

  def union(self, cluster1, cluster2):
    u = []
    for c in cluster1:
      u.append(c)
    for c in cluster2:
      if c not in u:
        u.append(c)
    return len(u)
    
  def intersection(self, cluster1, cluster2):
    i = []
    for c in cluster1:
      if c in cluster2:
        i.append(c)
    return len(i)
     
  def print_mappings_and_scores(self):
    print("Mappings from k-means clusters to agglomerative clusters\n")
    js_scores = []
    for k1 in self.kmeans_clusters.keys():
      max_j = -float('inf')
      for k2 in self.agg_clusters.keys():
        jss = self.intersection(self.kmeans_clusters[k1],self.agg_clusters[k2])/self.union(self.kmeans_clusters[k1],self.agg_clusters[k2]) 
        if jss>max_j:
          max_j = jss
          p = k1
          q = k2
      js_scores.append((p,q,max_j))
      print("Cluster",k1,"of kmeans is mapped to Cluster",q,"of agglomerative")

    print()
    print("\tJaccard Similarity Scores\n")
    for k1,k2,j in js_scores:
      print('Jaccard Similarity Score for the',k1,'->',k2,'mapping:',j)

#Main
if __name__ == "__main__":
  # Loading Dataset
  print("+++ Loading Dataset...\n")
  train_pd = pd.read_csv("COVID_1_unlabelled.csv")
  train_pd.drop(columns=['Unnamed: 0'], inplace = True)
  train_np = np.array(train_pd)

  # Z score normalization
  print("+++ Normalizing...\n")
  normalize(train_np)

  # Applying K-means
  print("+++ Applying K-means...\n")
  k_means = K_Means(train_np)
  g = k_means.final_centroids()
  final_clusters = k_means.get_clusters(g)
  kmeans_clusters = k_means.get_clusters_with_datapoint_indices(g)

  # Storing k-means clusters
  print("+++ Storing Clusters...\n")
  store_kmeans_clusters(kmeans_clusters)

  # Calculating silhouette coefficient for k = 3 (K-means)
  print("+++ Evaluating current model...\n")
  eval = Evaluate(k_means)
  print("The value of Silhouette coefficient for k = 3 is",eval.silhouette_coefficient())
  print()

  # Printing values of silhouette coefficient for k = 3, 4, 5, and 6 and finding the maximum
  print_best_k()
  print()

  # Applying agglomerative clustering
  print("+++ Applying agglomerative...\n")
  agg = Agglomerative(train_np,k = 3)
  g = agg.get_clusters()

  # preparing clusters with datapoint indices
  agg_clusters = dict()
  train_l = train_np.tolist()

  for k in g.keys():
    agg_clusters[k] = []
    for d in g[k]:
      l = d.tolist()
      agg_clusters[k].append(train_l.index(l))

  # Storing agglomerative clusters
  print("+++ Storing Clusters...\n")
  store_agg_clusters(agg_clusters)

  # printing jaccard mappings and scores
  print("+++ Computing Jaccard Similarity...\n")
  jaccard = Jaccard(kmeans_clusters, agg_clusters)
  jaccard.print_mappings_and_scores()
  print()
  print('Time Taken =',str(time.time() - start_time))