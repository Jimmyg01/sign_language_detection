Real-Time Sign Language Detection: Numbers 0-9  
This project implements a machine learning (CNN) model that translates hand gestures for numbers 0-9 into sign language in real time, with an associated confidence score.   


I built it using a Convolutional Neural Network (CNN) in PyTorch, the model processes live video input from the webcam and classifies the gestures with over 90% validation accuracy.  

Project Overview  
-Objective: Detect and interpret hand gestures for numbers (0-9) in real-time.  
-Data Collection: Automated pipeline captures 100 images per gesture from webcam input and stores them in labeled folders, ready for training.  
-Model Architecture: CNN with multiple convolutional and pooling layers, followed by fully connected layers to classify gestures.  
-Real-Time Processing: Model is deployed to recognize and display sign language gestures instantly with high responsiveness, though further accuracy improvements are possible with additional data.  


Technology  
-PyTorch: For building and training the CNN model.  
-OpenCV: For real-time video capture and image processing.  
-NumPy & PIL: For data handling, processing, and manipulation.  
-Scikit-Learn: For dataset splitting and model evaluation.  


Check out the code and more details in the repository!
