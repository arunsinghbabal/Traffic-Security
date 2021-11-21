# Traffic Security

# Aim of the project:
The project was aimed to modernize the highway system by implementing the following steps:
## 1. Vehicles classification and traffic frequency:
To classify the vehicles and provide the traffic frequency by obtaining individual vehicle counts over a period, which can be used for the road survey, automatic toll collection etc.
## 2. Vehicle tracking:
To provide vehicle location at a point of time by capturing vehicle image, licence plate image and slight details i.e., driving direction, unique ID number, capturing date and time, which can be used for effective tracking.
## 3. Saftey measures:
By categorizing the vehicles based on the provided flagged licence plates into suspicious and normal. The CSV file contains all the vehicle details including its licence plate number in text format and whether the vehicle is flagged (suspicious or not) for further preprocessing.

# Summary
![image](https://user-images.githubusercontent.com/93785299/142751217-c7e94de3-33c8-4985-b950-949448dc35b0.png)
# Project Implementation:

## Install dependencies:
First create an environment and install the dependencies listed in the **requirements.txt** file. <br />

>(base) Project_folder>**conda create -n "environment_name" python=3.8** # Create an environment <br />
>(base) Project_folder>**conda activate "environment_name"** # Activate the created environment <br />
>(environment) Project_folder>**pip install -r requirements.txt** # Install all the dependencies <br />

## Vehicle's classification and traffic frequency:
For only vehicle classification and their total count run the following commands:

>(base) Project_folder>**python main.py "VGG"** # To run on VGG16 CNN architecture  <br />
>
>(base) Project_folder>**python main.py "mobile"** # To run on MobileNetV2 CNN architecture  <br />

**Demo video**


## Vehicle tracking:
It will classify vehicles and count their number along with their images, licence plate and further details. To run, use the following commands:

>(base) Project_folder>**python main.py "Video+Licence plate" "VGG"** # To run on VGG16 CNN architecture  <br />
>
>(base) Project_folder>**Python main.py "Video+Licence plate" "mobile"** # To run on MobileNetV2 CNN architecture  <br />

**Demo video**

## Saftey measures:
After Vehicle's classification, traffic frequency and vehicle tracking, it will flag the vehicles and give the output in the csv format based on the provided details. To run, use the following commands:

>(base) Project_folder>**python main.py "Video+Licence plate+Text" "VGG"** # To run on VGG16 CNN architecture  <br />
**Demo video**

