import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from vame.util.auxiliary import read_config


class umap_visualization_silvia:
    def __init__(self,config):
       #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       #read all the necesary parameters
       self.model_name = cfg['model_name']
       self.n_cluster = cfg['n_cluster']
       self.parameterization = cfg['parameterization']
       self.project_path = cfg['project_path']
       self.file_exp = cfg['video_sets']
       file_1 = "latent_vector_" + self.file_exp[0] + ".npy"
       self.file_latent_vector = os.path.join(self.project_path,"results", self.file_exp[0], self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
       self.save_data = os.path.join(cfg['project_path'],"results",self.file_exp[0],self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
       self.file_labels = os.path.join(self.save_data,str(self.n_cluster)+'_km_label_'+ self.file_exp[0] + '.npy')
       self.umap_without_labels = os.path.join(self.save_data,str(self.n_cluster)+'_umap_without_label_'+ self.file_exp[0] + '.pdf')
       self.umap_with_labels = os.path.join(self.save_data,str(self.n_cluster)+'_umap_with_label_'+ self.file_exp[0] + '.pdf')

       
       
       #parameters for umap
       self.num_points = cfg['num_points']
       self.min_dist=cfg['min_dist']
       self.n_neighbors=cfg['n_neighbors']
       self.random_state = cfg['random_state']

    def  __call__(self,label = None):
        self.latent_vector = np.load(self.file_latent_vector)

        if self.num_points > self.latent_vector.shape[0]:
                self.num_points = self.latent_vector.shape[0]
        print("Embedding %d data points.." %self.num_points)

        embed = self.create_embedding()

        print("Visualizing %d data points.. " %self.num_points)
        if label == None:                    
            self.umap_vis(embed)
        elif label == "motif":
             motif_label = np.load(self.file_labels)
             self.umap_label_vis(embed,motif_label)
        

    #################################################################################    

    '''
    input: latent space
    output: embedding vectors
    '''
    def create_embedding(self):
         print("Compute embedding for file %s" % self.file_exp[0])
         reducer = umap.UMAP(n_components=2, min_dist = self.min_dist, n_neighbors = self.n_neighbors, 
                    random_state=self.random_state) 
         embed = reducer.fit_transform(self.latent_vector[:self.num_points,:])

         return embed
    
    '''
    input: embed vectors with number of points to plot
    output: plot without labels
    '''
    def umap_vis(self, embed):
         fig, ax = plt.subplots(figsize=(6,6))
         ax.scatter(embed[:self.num_points,0], embed[:self.num_points,1], s = 10, alpha = 0.6, c = 'blue', edgecolors = 'none')
         ax.set_aspect('equal', 'datalim')
         ax.set_xlabel("Embedding Dimension 1")
         ax.set_ylabel("Embedding Dimension 2")
         ax.set_title("2D Embedding Scatter Plot")

         ax.grid(False)

         fig.savefig(self.umap_without_labels, format = "pdf")

         plt.show()

    '''
    input: embed vectors with number of points to plot
    output: plot with labels
    '''
    def umap_label_vis(self,embed,label):
         fig, ax = plt.subplots(figsize=(7, 7))

         # Generate 28 distinct colors using tab20 + tab20b
         base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
         fixed_colors = base_colors[:28]   # take first 28 colors

        # Create a discrete colormap
         cmap = ListedColormap(fixed_colors)

         sc = ax.scatter(embed[:self.num_points, 0], embed[:self.num_points, 1], c=label[:self.num_points], cmap = cmap, s=10, alpha=0.7, edgecolors='none')

         # Add colorbar with cluster ticks
         cbar = plt.colorbar(sc, ax=ax, boundaries=np.arange(self.n_cluster+1)-0.5)
         cbar.set_ticks(np.arange(self.n_cluster))
         cbar.set_label("Cluster ID")

         ax.set_aspect('equal', 'datalim')
         ax.set_xlabel("Embedding Dimension 1")
         ax.set_ylabel("Embedding Dimension 2")
         ax.set_title("2D Embedding Scatter Plot")

         ax.grid(False)


         for cluster_id in range(self.n_cluster):
             # Get points belonging to this cluster
            cluster_points = embed[label[0] == cluster_id,:]
            if len(cluster_points) > 0:
              # Compute cluster centroid
              centroid = cluster_points.mean(axis=0)
              # Place text at centroid
              ax.text(centroid[0], centroid[1], str(cluster_id),
            fontsize=9, fontweight='bold',
            ha='center', va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

         fig.savefig(self.umap_with_labels, format = "pdf")

         plt.show()
