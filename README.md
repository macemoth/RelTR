Fork of the [yrcong/RelTR](https://github.com/yrcong/RelTR/) repository, optimised for production rather than evaluation. 


# Installation

## RelTR (CPU)

1. (Optional, but recommended) Create conda environment with `conda create -n reltr python=3.6` and activate it
2. Install PyTorch and PyVision with `pip install torch==1.6.0 torchvision==0.7.0 --extra-index-url https://download.pytorch.org/whl/cpu`
3. `pip install matplotlib scipy`
4. Download the [RelTR model](https://drive.google.com/open?id=1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD) and place it into `ckpt`

For help installing PyTorch, follow [PyTorch instructions](https://pytorch.org/get-started/locally/#supported-linux-distributions)

## Training/Evaluation on Visual Genome
If you want to **train/evaluate** RelTR on Visual Genome, you need a little more preparation:

a) Scipy (we used 1.5.2) and pycocotools are required. 

```shell
conda install scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

b) Download the annotations of [Visual Genome (in COCO-format)](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) and unzip it in the ```data/``` forder.

c) Download the the images of VG [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Unzip and place all images in a folder ```data/vg/images/```

d) Some widely-used evaluation code (**IoU**) need to be compiled... We will replace it with Pytorch code.

```shell
# compile the code computing box intersection
cd lib/fpn
sh make.sh
```

The directory structure looks like:

```shell
RelTR
| 
│
└───data
│   └───vg
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       |   images
└───datasets    
... 
```

# Usage

## Inference

Run

```shell
python inference.py --img_path $IMAGE_PATH --resume $MODEL_PATH --device cpu [--export_path graph.json]
```

Or, if you have a CUDA-capable device, replace `cpu` by `cuda`.


## Training
a) Train RelTR on Visual Genome on a single node with 8 GPUs (2 images per GPU):

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset vg --img_folder data/vg/images/ --batch_size 2 --output_dir ckpt
```

## Evaluation
b) Evaluate the pretrained RelTR on Visual Genome with a single GPU (1 image per GPU):

```shell
python main.py --dataset vg --img_folder data/vg/images/ --eval --batch_size 1 --resume ckpt/checkpoint0149.pth
```
