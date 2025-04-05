## scene_seg_trainer.py

Helper class for training SceneSeg neural network

## train_scene_seg.py

Main script for training SceneSeg neural network

## test_validate_scene_seg.py

Script to run SceneSeg neural network on full validation and test data and calculate key metrics

## scene_3d_trainer.py

Helper class for training Scene3D

## train_scene_3d.py

Main script for training Scene3D neural network

### Example usage

**Use random Scene3D weights with fixed pre-trained backbone from SceneSeg**

```bash
  python3 train_scene_3d.py -s /model_save_path -r /data_root_path -t /test_images_save_path -m /path_to_SceneSeg_network_weights.pth
```

**Load network from checkpoint saved during earlier stage of training**

```bash
  python3 train_scene_3d.py -s /model_save_path -r /data_root_path -t /test_images_save_path -l  -c /path_to_Scene3D_saved_network_weights.pth
```
### Parameters:

*-s , --model_save_root_path* : root path where pytorch checkpoint file should be saved

*-m , --pretrained_checkpoint_path* : path to SceneSeg weights file for pre-trained backbone

*-c , --checkpoint_path* : path to Scene3D weights file for training from saved checkpoint

*-r , --root* : path to folder where data training data is stored

*-t , --test_images_save_root_path* : root path where test images are stored

*-l , --load_from_save* : flag for whether model is being loaded from a Scene3D checkpoint file
