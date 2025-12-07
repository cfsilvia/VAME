import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import umap


from hmmlearn import hmm
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE

class cluster_latent_space_silvia:
    def __init__(self,config):
       #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       self.model_name = cfg['model_name']
       self.n_cluster = cfg['n_cluster']
       self.parameterization = cfg['parameterization']
       self.project_path = cfg['project_path']
       self.file_exp = cfg['video_sets']
       file_1 = "latent_vector_" + self.file_exp[0] + ".npy"
       self.file_latent_vector = os.path.join(self.project_path,"results", self.file_exp[0], self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
       self.save_data = os.path.join(cfg['project_path'],"results",self.file_exp[0],self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
       self.file_labels = os.path.join(self.save_data,str(self.n_cluster)+'_km_label_'+ self.file_exp[0] + '.npy')
       self.video_file = os.path.join(self.project_path, "videos", cfg['name_video'][0]) 
       self.cluster_start = cfg['time_window'] / 2
       self.data_file = os.path.join(self.project_path, "videos", "pose_estimation" ) 

    def  __call__(self,command):
        print('Pose segmentation for VAME model: %s \n' %self.model_name)
        match command:
            case "cluster":
                #load file
                 self.latent_vector_files = np.load(self.file_latent_vector)
                #cluster latent space 
                 labels, cluster_center = self.same_parameterization()
                 np.save(self.file_labels, labels)
            case "usage_motifs":
              self.labels = np.load(self.file_labels)
              motif_usage = self.get_motif_usage()
              self.find_best_number_cluster(motif_usage)
            case "find_motifs_on_the_movies":
                self.labels = np.load(self.file_labels)
                self.create_movies_with_same_motifs()
 ############################################################      

    '''
      clustering latent space
    '''
    def   same_parameterization(self):
      labels = []
      cluster_centers = []

      if self.parameterization == "kmeans":
        print("Using kmeans as parameterization!")
        kmeans = KMeans(init='k-means++', n_clusters=self.n_cluster, random_state=42, n_init=20).fit(self.latent_vector_files)
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(self.latent_vector_files)
      elif self.parameterization == "hmm":
        print("Using a HMM as parameterization!")
        hmm_model = hmm.GaussianHMM(n_components=self.n_cluster, covariance_type="full", n_iter=100)
        hmm_model.fit(self.latent_vector_files)
        label = hmm_model.predict(self.latent_vector_files)
      
      labels.append(label)
      if self.parameterization == "kmeans":
            cluster_centers.append(clust_center)
      return labels, cluster_centers

    '''
   find the index where it begins a new motifs
  '''
    @staticmethod
    def consecutive(data, stepsize=1):
     data = data[:]
     return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


    def get_motif_usage(self):
      motif_usage = np.unique(self.labels, return_counts = True) #say motif_usage[0] the clusters, motif_usage[1] how many motifs appear each time
      motif_usage = motif_usage[1] #number of times the motif appears

      return motif_usage
    
    '''
    Find the best number of clusters all which are given 1% 
    '''
    def find_best_number_cluster(self, motif_usage):
       sorted_arr = np.sort(motif_usage)[::-1]
       percent = sorted_arr / sorted_arr.sum()*100

       fig,ax = plt.subplots(figsize=(6,4))
       ax.plot(percent)
       ax.axhline(1, linestyle= '--', color = "red" )
       ax.set_ylabel("usage_motifs %")

       plt.show()

    '''
    create movies with the same motif
    '''
    def create_movies_with_same_motifs(self):
       #settings
       
       file_path_output = os.path.join(self.data_file,(self.file_exp[0] + "_information_after_clustering.xlsx"))
       csv_file = os.path.join(self.data_file, (self.file_exp[0] + ".csv"))
       df_csv = pd.read_csv(csv_file)

       df_new = pd.DataFrame(columns = [*df_csv.columns, "motif", "#event", "#frame"])
       #read labels
       motifs = np.unique(self.labels)
       
       for m in motifs:
             #read movie 
             capture = cv2.VideoCapture(self.video_file)
             #vid information
             if capture.isOpened():
                width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                print('fps:', fps)
                #extract frames for a given motifs
                frames = np.where(self.labels.squeeze() == m)[0]
                
                self.write_movie(m,capture,frames, width, height, fps)
                df_new = self.write_data(df_new,m,frames, df_csv)
               
                df_new.to_excel(file_path_output, index = False)  
       df_new.to_excel(file_path_output, index = False)     


    '''
    write a movie
    '''
    def write_movie(self, m, capture, frames_numbers,width, height, fps):
       #create mp4 writer
       
       output_path = os.path.join(self.project_path, "videos", self.file_exp[0] + "_snapshot_" + str(m) + "_.mp4")

       
      #  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      #  writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
       writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))
       #
       differences = np.diff(frames_numbers)
       number_serie = 0
       for count, f in enumerate(frames_numbers):
          capture.set(cv2.CAP_PROP_POS_FRAMES, self.cluster_start + f)
          ret, frame = capture.read()
          if ret:
            #writer.write(frame)
            #add text
            if (count > 0) and differences[count-1] > 1:
                number_serie += 1
                text = str(int(number_serie)) + ' New Event  ' + str(self.cluster_start + f ) + str(number_serie) 
            else:
                text = str(int(number_serie)) + ' Frame  ' + str(self.cluster_start + f )
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 8
            color = (0, 0, 255)  # Red text (BGR format)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            frame_height, frame_width = frame.shape[:2]
            position = (
                (frame_width - text_width) // 2,
                (frame_height + text_height) // 2
                )
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            writer.write(frame)

          else:
            print(f"Could not extract frame {f}")

       capture.release()
       writer.release()
       print(f"Created video: {output_path}")

    '''
     write data
    '''
    def write_data(self,df_new,m,frames_numbers, df_csv):
       differences = np.diff(frames_numbers)
       number_serie = 0
       for count, f in enumerate(frames_numbers):
            if (count > 0) and differences[count-1] > 1:
                number_serie += 1
            df_aux = df_csv.iloc[[f]]
            df_aux['motif'] = m
            df_aux['#event'] = number_serie
            df_aux['#frame'] = f

            df_new = pd.concat([df_new, df_aux], ignore_index=True)
           
       return df_new

    
   