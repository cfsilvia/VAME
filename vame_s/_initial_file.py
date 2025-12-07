import sys
#sys.path.insert(0, r"F:\SilviaData\ScriptOnGithub\VAME")
sys.path.insert(0, r"D:\Silvia\ScriptOnGithub\VAME")
import vame_silvia
from create_file_for_vame import create_file_for_vame
from etoghram import etoghram





def main_menu(data):
    choice = data['choice']
    match choice:
        case '1': #create file 
            obj_1 = create_file_for_vame(data['original_data'], data['sheetname'], data['upper_tube'], data['lower_tube'])
            obj_1()
        case '2':
            # Initialize your project
            config = vame.init_new_project(project=user_data['project'], videos=data['videos'], working_directory=data['working_directory'], videotype='.mp4')
        case '3':
            # you can use the following to convert your .csv to a .npy array, ready to train vame on it
             vame.csv_to_numpy(data['config'])
        case '4':
             # # create the training set for the VAME model
             vame.create_trainset(data['config'], check_parameter=False)
        case '5':
             # # Train VAME:
             vame.train_model(data['config'])
        case '6':
             # # Evaluate model
             vame.evaluate_model(data['config'])
        case '7':
             # # Segment motifs/pose
             vame.pose_segmentation(data['config'])
        case '8':
            #Create motif videos to get insights about the fine grained poses
            vame.motif_videos(data['config'], videoType='.mp4')
        case '9':
            #Create etograms of the motifs
            input_file = user_data['labels_file']
            fps = user_data['fps']  # adjust to your recording's frame rate
            output_csv_path = user_data['file_etogrhams']
            offset = 30/2  # depends on your window size (usually half of it)
            etoghram(input_file, output_csv_path,fps, offset)
        case '10':
            #general analysis on the latent space
            obj = vame.cluster_latent_space_silvia(data['config'])
            obj("find_motifs_on_the_movies") #commands to add : "cluster", "usage_motifs", "find_motifs_on_the_movies"
        case '11':
            #do umap over the latent space
             obj = vame.umap_visualization_silvia(data['config'])
             obj("motif") #argument could be: blank nothing or "motif"
            

        case _:
             return "Invalid option"

 

if __name__ == "__main__":
    
    user_data = {}
    user_data['choice'] = '11'
    user_data['working_directory'] = 'D:/Silvia/Data/Data_for_vame/BMR10/VAME/' #'U:/Users/Ruthi/2025/BMR10/VAME/'
    user_data['original_data'] = 'D:/Silvia/Data/Data_for_vame/BMR10/VAME/BMR10_with_landmarks_left.xlsx' #'U:/Users/Ruthi/2025/BMR10/VAME/BMR10_with_landmarks_left.xlsx'
    user_data['sheetname'] = 'BMR'
    user_data['upper_tube'] = 1257
    user_data['lower_tube'] = 1455
    user_data['project']='BMR10-VAME-Project'
    user_data['videos'] = ['D:/Silvia/Data/Data_for_vame/BMR10/VAME/BMR10_with_landmarks_left.csv'] #['U:/Users/Ruthi/2025/BMR10/VAME/BMR10_with_landmarks_left.csv']
    user_data['config'] = 'D:/Silvia/Data/Data_for_vame/BMR10/VAME/' + 'BMR10-VAME-Project-Nov30-2025' + '/config.yaml' #'U:/Users/Ruthi/2025/BMR10/VAME/' + 'BMR10-VAME-Project-Jul24-2025' + '/config.yaml'
    user_data['labels_file'] = r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Nov19-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\10_km_label_BMR10_with_landmarks_left.npy" #r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Jul24-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\10_km_label_BMR10_with_landmarks_left.npy"
    user_data['fps'] = 24.00
    user_data['file_etogrhams'] = r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Nov19-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\ethogram_aligned.csv" #r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Jul24-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\ethogram_aligned.csv"
    
    # user_data['directory_results'] = r"D:\Silvia\Data\Data_for_vame\BMR10\VAME\BMR10-VAME-Project-Nov25-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\"
    # user_data['number_clusters'] = 10
    # user_data['name_data'] = "BMR10_with_landmarks_left"

    main_menu(user_data)