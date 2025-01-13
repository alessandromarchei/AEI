import json
import os

# Load JSON data
with open('drivable_path.json', 'r') as file:
    data = json.load(file)
data = data['data']
# Directory where images are stored
image_directory = 'images'
vis_directory = 'visualization'
mask_directory = 'segmentation'

# Filter out elements with missing image files
new_data = []
for item in data:
    img_name = list(item.keys())[0]
    if os.path.isfile(os.path.join(vis_directory, img_name+'.png')):
        new_data.append(item)
    else:
        os.remove(os.path.join(image_directory, img_name+'.png'))
        os.remove(os.path.join(mask_directory, img_name+'.png'))

j_data = {"data":new_data}
# Save the filtered data back to the JSON file
with open('drivable_path.json', 'w') as file:
    json.dump(j_data, file, indent=4)

print("Filtered JSON data saved.")