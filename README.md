Project Documentation: Mark Predictor AI (v2.1)

Project Title: Mark Predictor AI

Version: 2.1

Date: July 19, 2025

Author: [Your Name]

1. Project Summary

Mark Predictor AI (v2.1) is a web application designed to predict student marks using a Deep Neural Network (DNN). The primary goal of this tool is to help students understand the impact of their daily time allocation—specifically hours spent on studying, sleeping, and playing—on their academic performance, thereby enabling them to create more effective personal schedules.

2. Key Features

Deep Learning Model: Utilizes a DNN built with PyTorch to capture complex relationships between a student's daily activities and their marks.

Hyperparameter Tuning: Employs Optuna to automatically find the best hyperparameters for the model, ensuring optimal performance.

Web Interface: A simple front-end built with HTML and powered by a Flask backend, allowing users to easily input their data and receive instant predictions.

Data-Driven: Trained on an expanded dataset of 5,000 samples to provide more robust and reliable predictions.

3. Enhancements in Version 2.1

This version is a significant upgrade from its predecessor with several key improvements:

Larger Dataset: The model is now trained on a dataset of 5,000 entries, a major increase that leads to improved accuracy.

Learning Rate Scheduler: A StepLR scheduler has been added to the training process, which helps the model converge more effectively.

Enhanced Frontend: The user interface now includes fields for Student ID, Student Name, and Student Course for a more complete user experience.

4. Technology Stack

Backend: Flask

Machine Learning Framework: PyTorch

Hyperparameter Optimization: Optuna

Frontend: HTML

Programming Language: Python
