# How to use the process_cmu_roadwork_data.py file

Assume the path to folder with training label is "../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/train/", this is assigned to -tr
Assume the path to the folder with validation label is "../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/val/", this is assigned to -va
Assume the processed labels will be stored in a folder named "label", this is assigned to -lbs
Assume the processed images will be stored in a folder named "image", this is assigned to ims


python process_cmu_roadwork_data.py -tr ../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/train/ -va ../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/val/ -lbs label -ims image