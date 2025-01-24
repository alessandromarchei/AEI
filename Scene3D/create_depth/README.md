
## Scene3D Dataset
To train the Scene3D network, a custom dataset was created using a number of open-source, publicly available datasets which capture scene depth through simulation, LIDAR or a combination of LIDAR and Stereo Vision.

The Scene3D dataset includes RGB images, ground truth depth maps, ground truth height maps, and validity masks to mark valid vs invalid raw depth readings.

![SuperDepth_Data](../../Diagrams/SuperDepth_Data.jpg)

The datasets used to train SuperDepth include:

- [Argoverse](https://www.argoverse.org/av1.html#stereo-link)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [UrbanSyn](https://www.urbansyn.org)

## common
Contains helper classes used to create ground truth data