# Monocular 3D Object Reconstruction with GAN Inversion (ECCV 2022)

This paper presents a novel GAN Inversion framework for single view 3D object reconstruction.

* Project page: [link](https://www.mmlab-ntu.com/project/meshinversion/)
* Paper: [link](https://arxiv.org/abs/2207.10061)
* Youtube: [link](https://www.youtube.com/watch?v=13QfxbZqmvM)


## Setup
Install environment:
```
conda env create -f env.yml
conda activate mesh_inv
```
Install [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) (tested on commit [e7e5131](https://github.com/NVIDIAGameWorks/kaolin/tree/e7e513173bd4159ae45be6b3e156a3ad156a3eb9)).

Download the [pretrained model](https://drive.google.com/file/d/1TeE_c0V3lWd5y5Ine4Gmesc2O4cfIH9S/view?usp=sharing) and place it under `checkpoints_gan/pretrained`.  Download the CUB dataset [CUB_200_2011](http://www.vision.caltech.edu/datasets/cub_200_2011/), [cache](https://drive.google.com/file/d/11PPf-obl-eakPElU6ghcgkje8S8hwFrT/view?usp=sharing), [predicted_mask](https://drive.google.com/file/d/1L-pbvxb6jL7fUEyFPPRgXHNHsK2U01qo/view?usp=sharing), and [PseudoGT](https://drive.google.com/file/d/1wCfVDRx_8DJzfP7aYBX0AQXs4LYxX4rI/view?usp=sharing) for ConvMesh GAN training, and place them under `datasets/cub/`. Alternatively, you can obtained your own predicted mask by PointRend, and you can obtain your own PseudoGT following [ConvMesh](https://github.com/dariopavllo/convmesh). 

```
- datasets
  - cub
    - CUB_200_2011
    - cache
    - predicted_mask
    - pseudogt_512x512
```

## Reconstruction
The reconstruction results of the test split is obtained through GAN inversion.
```
python run_inversion.py --name author_released --checkpoint_dir pretrained 
```

## Evaluation
Evaluation results can be obtained upon GAN inversion.
```
python run_evaluation.py --name author_released --eval_option IoU
python run_evaluation.py --name author_released --eval_option FID_1
python run_evaluation.py --name author_released --eval_option FID_12
python run_evaluation.py --name author_released --eval_option FID_10
```

## Pretraining
You can also pretrain your own GAN from scratch.
```
python run_pretraining.py --name self_train --gpu_ids 0,1,2,3 --epochs 600
```

## Acknowledgement
The code is in part built on [ConvMesh](https://github.com/dariopavllo/convmesh), [ShapeInversion](https://github.com/XingangPan/deep-generative-prior) and [CMR](https://github.com/chenyuntc/cmr). Besides, Chamfer Distance is borrowed from [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), which is included in the `lib/external` folder for convenience.

## Citation
```  
@inproceedings{zhang2022monocular,
    title = {Monocular 3D Object Reconstruction with GAN Inversion},
    author = {Zhang, Junzhe and Ren, Daxuan and Cai, Zhongang and Yeo, Chai Kiat and Dai, Bo and Loy, Chen Change},
    booktitle = {ECCV},
    year = {2022}}
```
