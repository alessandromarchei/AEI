# The process_roadwork.py file 
The aim of the process_roadwork_data.py file is to create a folder of images and a corresponding folder of labels. A label will have a value of 1 where the following original Class-Label values are present 
	Class-Label     =   [index2, index1, index0]
	Barrier........	=   [6, 6, 6]
	Barricade......	=   [7, 7, 7]
	Fence..........	=   [8, 8, 8]
	work_vehicle... =   [10, 10, 10]
	policeofficer..	=   [11, 11, 11]
	worker.........	=   [12, 12, 12]
	cone...........	=   [13, 13, 13]
	drum...........	=   [14, 14, 14]
	vertical_panel.	=   [15, 15, 15]
	tubular_marker.	=   [16, 16, 16]
	work_equipement	=   [17, 17, 17]
	arrow_board....	=   [18, 18, 18]
	ttc_sign......	=   [19, 19, 19]
     
and a value of 0 otherwise (for the Class-Label values below).
	Class-Label     =   [index2, index1, index0]
	Road........... =   [1, 1, 1]
	Sidewalk.......	=   [2, 2, 2]
	Bike Lane...... =   [3, 3, 3]
	Off_Road.......	=   [4, 4, 4]
	RoadSide.......	=   [5, 5, 5]
	Police_Vehicle. =   [9, 9, 9]


What this means is that, in the original label, we had different label values or identities for 19 different class, but in the new label, some of these different identities/class will have a value of 1 while others will have a value of zero. 

The new labels and images will also be processed to have the standard dimension of 1080 by 1920 each.  

# How to use the process_roadwork.py file
Assume the path to the folder with the original CMU RoadWork training label is "../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/train/", this is assigned to -tr in the python code
Assume the path to the folder with the original CMU RoadWork validation label is "../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/val/", this is assigned to -va in the python code
Assume the processed labels will be stored in a folder named "label", this is assigned to -lbs in the python code
Assume the processed images will be stored in a folder named "image", this is assigned to ims in the python code

Then execute the code below
python process_roadwork.py -tr ../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/train/ -va ../../../../Data/Data_May_5th_2025/cmu/sem_seg_labels/gtFine/val/ -lbs label -ims image