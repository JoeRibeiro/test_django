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

.py file usage:
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\dump_frames.py"
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskRCNN.py"
python -m image_labelling_tool.flask_labeller --images_dir "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\stills"