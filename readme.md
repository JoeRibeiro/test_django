This repository was an exploration to train a Mask-RCNN model on a few annotations of fish which I labelled personally in the django labeller made by the University of East Anglia, it was not funded under any specific project and consequently I am leaving it open.


Set up environment with
conda env create -f environment.yml

Save environment with
conda env export > environment.yml


Step-by-step build (I am doubtful a .yml will install properly):
conda create --name djangoenv python=3.7 numpy=1.19.5 imutils=0.5.4
conda activate djangoenv
pip install django-labeller opencv-python
# At this point, django labeller works
# Now we want maskrcnn
pip install -r "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskrcnn\Mask_RCNN\requirements.txt"
cd "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskrcnn\Mask_RCNN\"
python setup.py install
# For maskcsnn, Coco needs to be the windows version, the default is not supported and you might need to install visual studio build tools
pip install pycocotools-windows
# Also maskRCNN is breaking without:
pip install h5py==2.10.0
pip install skikit-image=0.16.2

In the cloned_djangoenv I went further:
# To use the GPU, not the CPU:
C:\Users\JR13\AppData\Local\miniforge3\envs\cloned_djangoenv\python.exe -m pip install --upgrade setuptools pip wheel
C:\Users\JR13\AppData\Local\miniforge3\envs\cloned_djangoenv\python.exe -m pip install nvidia-pyindex
py -m pip install tensorflow-gpu==1.15.2

# install gpu version of tensorflow:
Go to https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal, download and install executable
Likewise with:
https://developer.nvidia.com/rdp/cudnn-archive
and Download cuDNN v7.6.0 (May 20, 2019), for CUDA 10.0
EXTRACT AND PASTE IT into EACH OF THE RELEVANT DIRECTORIES WITHIN your installation directory C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0, see https://stackoverflow.com/questions/48698536/tensorflow-gpu-import-tensorflow-importerror-could-not-find-cudnn64-7-dll

... This runs predcitions on GPU, but the result is nonsense. The model will not train on GPU.
I think I should abandon using tensorflow 1.15 and the 6 year old maskRCNN model and rewrite this entire readme. There seems to be a more recently maintained version here: https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0
For now though, I should work in djangoenv and not cloned_djangoenv and accept it runs slowly.

# SORT requirements
pip install filterpy==1.4.5
pip install lap==0.4.0


.py file usage:
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\dump_frames.py"
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskRCNN_for_fish.py"
python -m image_labelling_tool.flask_labeller --images_dir "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\stills"
