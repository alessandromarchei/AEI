#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import torch.nn as nn
from pytorch_model_summary import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork

class SuperDepthUpstream(nn.Module):
    def __init__(self, pretrainedModel):
        super(SuperDepthUpstream, self).__init__()

        self.pretrainedBackBone = pretrainedModel.Backbone
        for param in self.pretrainedBackBone.parameters():
            param.requires_grad = False

        self.pretrainedContext = pretrainedModel.SceneContext
        for param in self.pretrainedContext.parameters():
            param.requires_grad = False

    def forward(self, image):
        features = self.pretrainedBackBone(image)
        deep_features = features[4]
        context = self.pretrainedContext(deep_features)
        return features, context
       

def main(): 

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')


    
    # Load pre-trained weights
    sceneSegNetwork = SceneSegNetwork()
    root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'
    sceneSegNetwork.load_state_dict(torch.load \
            (pretrained_checkpoint_path, weights_only=True, map_location=device))
    
    # Instantiate Model
    superDepthUpstream = SuperDepthUpstream(sceneSegNetwork)
    print(summary(superDepthUpstream, torch.zeros((1, 3, 320, 640)), show_input=True))
    
    # Model to device
    superDepthUpstream = superDepthUpstream.to(device)

    # Random input
    input_image_filepath = '/mnt/media/SuperDepth/UrbanSyn/image/10.png'
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Image loader
    image_loader = transforms.Compose(
        [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
    )

    image_tensor = image_loader(image_pil)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Generated outputs
    features, context = superDepthUpstream(image_tensor)

    # Getting outputs
    features = features[0].squeeze(0).cpu().detach()
    features = features.permute(1, 2, 0)
    context = context.squeeze(0).cpu().detach()
    context = context.permute(1, 2, 0)

    # Output
    plt.figure()
    plt.imshow(image_pil)    
    plt.figure()
    plt.imshow(features[:,:,20])
    plt.figure()
    plt.imshow(context[:,:,20])


if __name__ == '__main__':
    main()
#%%