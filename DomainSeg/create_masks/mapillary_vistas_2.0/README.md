
# The process_mapillary_vistas.py"
The goal of the process_mapillary_vistas.py file is to create a folder of images and a corresponding folder of labels. A label will have a value of 1 where the following original class label values are present 
	object--traffic-cone             [210, 60, 60]
	construction--barrier--temporary [250, 170, 35] 
and 0 otherwise.
The new labels and images will also be processed to have the standard dimension of 1080 by 1920 each. 

There are lots of images without the needed label class which can affect training efficiency. Because of this, a function called "get_needed_name_of_image_with_label" is developed with the aim of determining the names of images with the desired label class (object--traffic-cone and construction--barrier--temporary). After which these names will be used to select the images and labels that will be processed and stored in the folder of images and labels created. Because each pixel in all label-images in the folder needs to be examined, a GPU is needed to speed up operations. Without GPU, the task of examining all label-images for the desired label class could take more than a day. With GPU, it can take some minutes/an hour


# How to use the file "process_mapillary_vistas.py"
Assume the path to the folder with the original Mapillary Vistas RoadWork training label is "../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/", this is assigned to -trlb in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork training image is "../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/images/", this is assigned to -trim in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork validation label is
"../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/", this is assigned to -valb in the python code

Assume the path to the folder with the original Mapillary Vistas RoadWork validation image is
"../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/images/", this is assigned to -vaim in the python code

Assume the processed labels will be stored in a folder named "label", this is assigned to -lbs in the python code
Assume the processed images will be stored in a folder named "image", this is assigned to -ims in the python code

then execute the code below
python process_mapillary_vistas.py -trlb ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/v2.0/labels/ -trim ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/training/images/  -valb ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/v2.0/labels/ -vaim ../../../../Data/Data_May_5th_2025/Mapillary_Vistas/validation/images/ -lbs label -ims image


