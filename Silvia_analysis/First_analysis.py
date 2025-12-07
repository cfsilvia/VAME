import numpy as np


class First_analysis:
    def __init__(self, directory, number_clusters,  name_data):
        self.directory = directory


    def __call__(self):
        self.load_data()


    '''
    load numpy data
    '''
    def load_data():
        np.load()