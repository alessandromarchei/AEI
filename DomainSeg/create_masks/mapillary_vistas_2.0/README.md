# Their are two files (1) extract_relevant_file_names_MP.py (2) get_store_MP_RW_data.py 


# For the file "extract_relevant_file_names_MP.py"
The goal of this file is to extract the names of all files in the folder with the required label class. 
There are lots of images without the needed label class which can affect training efficiency.
Because each pixel in all images in the folder needs to be examined, a GPU is needed to speed up operations.
Without GPU, it could take more than a day
With GPU, it can take some minutes/an hour
The label classes of interest are object--traffic-cone [210, 60, 60] and construction--barrier--temporary [250, 170, 35]
The output of the "extract_relevant_file_names_MP.py" are two csv files which are 'train_address_modified.csv' and 'val_address_modified.csv'. These csv files have the names of the images/labels with the desired label classes  
if you already have these csv files, then there is no need to execute the "extract_relevant_file_names_MP.py" 


# How to use the file extract_relevant_file_names_MP.py
Assume the path to the folder with the original Mapillary Vistas RoadWork training label is "../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/", this is assigned to -tr in the python code
Assume the path to the folder with the original Mapillary Vistas RoadWork validation label is
"../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/", this is assigned to -va in the python code

then execute the code below
python extract_relevant_file_names_MP.py -tr ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/ -va ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/


# For the file "get_store_MP_RW_data.py"
The goal of the get_store_MP_RW_data.py file is to create a folder of images and a corresponding folder of labels. A label will have a value of 1 where the following original class label values are present 
	object--traffic-cone             [210, 60, 60]
	construction--barrier--temporary [250, 170, 35] 
and 0 otherwise.
The new labels and images will also be processed to have the standard dimension of 1080 by 1920 each.  

# How to use the file "get_store_MP_RW_data.py"
Assume the path to the folder with the original Mapillary Vistas RoadWork training label is "../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/", this is assigned to -trlb in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork training image is "../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/images/", this is assigned to -trim in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork validation label is
"../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/", this is assigned to -valb in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork validation image is
"../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/images/", this is assigned to -vaim in the python code

Assume the processed labels will be stored in a folder named "label", this is assigned to -lbs in the python code
Assume the processed images will be stored in a folder named "image", this is assigned to ims in the python code

then execute the code below
python get_store_MP_RW_data.py -trlb ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/ -trim ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/images/  -valb ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/ -vaim ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/images/ -lbs label -ims image


