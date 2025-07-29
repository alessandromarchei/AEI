# The process\_roadwork.py file

The aim of the process\_roadwork.py file is to create a folder of images and a corresponding folder of Binary labels. A label will have a value of 255 where the following original Class-Label values are present
cone
drum
vertical\_panel
tubular\_marker

and a value of 0 otherwise.

To perform the processing, Create a folder that would house the original unprocessed images and labels from the CMU RoadWork Website (https://kilthub.cmu.edu/articles/dataset/ROADWork\_Data/26093197?file=47217583). Then all images from the images.zip folder of the CMU website should be stored in a folder named images in the created folder. Also, all labels from the zip folders sem\_seg\_labels.zip/gtFine/train and sem\_seg\_labels.zip/gtFine/val of the CMU website should be stored in a folder named gtFine.

For example, if you created a folder called cmu\_x, then it will have two sub folders, one called images which will have all the images and the other folder called gtFine which will have all the labels from the CMU Website.



To execute the code,
Assume the relative path to the cmu\_x folder with the images/labels is "../../../../Data/Data\_May\_5th\_2025/cmu\_x/", this will be assigned to the variable -d in the python code.
The processed labels will be stored in a folder named "label", and the processed images will be stored in a folder named "image"; both folders (label and image) will be stored in a folder (relative path) name assigned to the variable -s. Assume we assign the variable -s as save;

Then execute the code below
python process\_roadwork.py -d ../../../../Data/Data\_May\_5th\_2025/cmu\_x/  -s save/



A processed version of the data can be found in the Kaggle link https://www.kaggle.com/datasets/austinosas/cmu-road-work-processed-data





