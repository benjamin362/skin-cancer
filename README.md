# skin-cancer
ISIC 2018 Challenge: Skin Lesion Analysis Towards Melanoma Detection


# Task 1
(Files are in segmentation folder)

Task one is to predicit a segmentation mask which covers the entire mole. 
Two different Unet's (small_Unet.py and big_Unet.py) have been implemented and trained with different loss functions. The best results where archieved with the large Unet and a mixed BCE and Dice Loss

# Task 2

Task two is to predict a feature mask for the feature "pigment network". The results that I show in the presentation are performed with Task2.ipynb, whereas Task2_initial_trial.ipynb is the implementation of U-Net with pretrained InceptionResNetV2 used as encoder that we decided not to use in the end because it was too heavy.

# Task 3

Task 3 is to predict the type of disease. There are seven possible options. How the images were loaded, training, validation and test including the cnn is shown in p.py. The file also shows different models that were implemented as comments. The best results were accomplished with the one in p.py.
