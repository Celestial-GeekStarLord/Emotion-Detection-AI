# Emotion-Detection-AI
Detect emotion using computer vision

Download DataSets from https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
Install requirements.txt in virtual environment & train and deploy model


Feature,What to change,Why?
Layers,         Add double Conv2D layers per block,Better feature extraction.
Augmentation,   Add RandomFlip and Rotation,Prevents the model from memorizing specific pixels.
Optimizer,      Add ReduceLROnPlateau,"Prevents ""overshooting"" the best accuracy."
Dropout,        Gradually increase (0.2 to 0.5),Forces the model to learn redundant features.