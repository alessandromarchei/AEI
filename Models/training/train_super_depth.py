#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_model_summary import summary
from PIL import Image
import sys
sys.path.append('..')
from model_components.super_depth_network import SuperDepthNetwork
from model_components.scene_seg_network import SceneSegNetwork


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # Load pre-trained weights
    sceneSegNetwork = SceneSegNetwork()
    root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'
    sceneSegNetwork.load_state_dict(torch.load \
        (pretrained_checkpoint_path, weights_only=True, map_location=device))
    
    # Instantiate Model with pre-trained weights
    model = SuperDepthNetwork(sceneSegNetwork)
    print(summary(model, torch.zeros((1, 3, 320, 640)), show_input=True))
    model = model.to(device)

    # Random input
    input_image_filepath = '/mnt/media/SuperDepth/UrbanSyn/image/10.png'
    image = Image.open(input_image_filepath)
    image = image.resize((640, 320))

    # Image loader
    image_loader = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
    )

    image_tensor = image_loader(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    prediction = model(image_tensor)

    prediction = prediction.squeeze(0).cpu().detach()
    prediction = prediction.permute(1, 2, 0)

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(prediction)

    
if __name__ == '__main__':
    main()
# %%
