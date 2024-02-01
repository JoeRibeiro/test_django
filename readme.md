Set up environment with
conda env create -f environment.yml

Save environment with
conda env export > environment.yml


Step-by-step build:
conda create --name djangoenv python=3.7 numpy=1.19.5 imutils=0.5.4
conda activate djangoenv
pip install django-labeller opencv-python
# At this point, django labeller works
pip install -r "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskrcnn\Mask_RCNN\requirements.txt"
cd "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskrcnn\Mask_RCNN\"
python setup.py install
pip install coco


.py file usage:
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\dump_frames.py"
python "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\maskRCNN.py"
python -m image_labelling_tool.flask_labeller --images_dir "C:\Users\JR13\OneDrive - CEFAS\My onedrive documents\test_django\stills"