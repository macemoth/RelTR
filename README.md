Fork of the [yrcong/RelTR](https://github.com/yrcong/RelTR/) repository, optimised for production rather than evaluation. 


# Installation

## RelTR (CPU)

1. (Optional, but recommended) Create conda environment with `conda create -n reltr python=3.6` and activate it
2. Install PyTorch and PyVision with `pip install torch==1.6.0 torchvision==0.7.0 --extra-index-url https://download.pytorch.org/whl/cpu`
3. `pip3 install -r requirements.txt`
4. Download the [RelTR model](https://drive.google.com/open?id=1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD) and place it into `ckpt`

For help installing PyTorch, follow [PyTorch instructions](https://pytorch.org/get-started/locally/#supported-linux-distributions)

# Usage

## Inference

To produce a scene graph from an image, run

```shell
python3 mkgraph.py --img_path $IMAGE_PATH --resume $MODEL_PATH --device cpu [--export_path graph.json]
```

Or, if you have a CUDA-capable device, replace `cpu` by `cuda`.

## Scene verbalisation

Set an environment variable `OPENAI_API_KEY=<YOUR API KEY>` with your OpenAI key.

````shell
pip3 install openai flask
python server.py
````

