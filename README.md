# Image-Classification
Footwear Type and Color Identification

## Dependencies
python - 2.7
keras - 2.0
tensorflow - 1.0
sklearn - 0.18

## Report
Report for the project is saved as report.docx

## Executable code
Executable code is in Prediction folder. The make_predicitions notebook can be referred on how predictions are being made.
Instructions to execute the code are mentioned in readme.md in the precition folder and also in the python(make_predictions.py) file.
Refer to the comments for step be step instructions and download the necessary data and classifiers(pickled) files.

## Footwear Type Folder 
Contains jupyter notebooks for:
- saving images 
- Fine Tuning using CNN
- Extracting features 
- Applying PCA transformation
- Training type_classifier using Support Vector Machines
- Computing validation accuracy(31%)
Note:
The files resulting as output on running these notebooks are not included as the prediction code can be executed without them. Also their size are very large.

## Footwear Color Folder:
Notebooks for training color_classifier 
- All the notebooks are somewhat similar to the notebooks in Footwear Type folder with few changes
- Validation and fine tuning notebooks are not present as fine tuning fails on small images and validation set was not built due to size constraints

Note:
Comments in the notebooks can be referred to learn more about how classifiers were trained
