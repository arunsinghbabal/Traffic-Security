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
![image](https://user-images.githubusercontent.com/93785299/142752574-f2ce83d0-def3-4eac-afd3-203b19624525.png)

# Project Implementation:

## Install dependencies:
First create an environment and install the dependencies listed in the **requirements.txt** file. <br />

>(base) Project_folder>**conda create -n "environment_name" python=3.8** # Create an environment <br />
>(base) Project_folder>**conda activate "environment_name"** # Activate the created environment <br />
>(environment) Project_folder>**pip install -r requirements.txt** # Install all the dependencies <br />

## Vehicle's classification and traffic frequency:
For only vehicle classification and their total count run the following commands:
<br/>
>(base) Project_folder>**python main.py"** # To run on **VGG16** CNN architecture  <br />
**or**<br/>
>(base) Project_folder>**python main.py "VGG"** # To run on **VGG16** CNN architecture  <br />

**Demo video**

https://user-images.githubusercontent.com/93785299/142752542-8ab36bdd-ef0d-4a0e-8231-f64b232ef345.mp4

<br />

**Output images**
 <br />
**1. Vehicle Image**
<br/>
![car ID_No 2 is going in direction up on 2021-11-21 at 06_17_42_816463](https://user-images.githubusercontent.com/93785299/142752869-49aa7d68-78c7-409f-bc47-cc17c0113d4b.png)
<br />


## Vehicle tracking:
It will classify vehicles and count their number along with their images, licence plate and further details. To run, use the following commands:
<br/>
>(base) Project_folder>**python main.py "Video+Licence plate"** # To run on **VGG16** CNN architecture  <br />
**or**<br/>
>(base) Project_folder>**python main.py "Video+Licence plate" "VGG"** # To run on **VGG16** CNN architecture  <br />
**or**<br/>
>(base) Project_folder>**Python main.py "Video+Licence plate" "mobile"** # To run on **MobileNetV2** CNN architecture  <br />

**Demo video**

https://user-images.githubusercontent.com/93785299/142752542-8ab36bdd-ef0d-4a0e-8231-f64b232ef345.mp4

<br />

**Output images**
 <br />
**1. Vehicle Image**
<br/>
![car ID_No 2 is going in direction up on 2021-11-21 at 06_17_42_816463](https://user-images.githubusercontent.com/93785299/142752869-49aa7d68-78c7-409f-bc47-cc17c0113d4b.png)
<br />
**2. Licence plate image**
<br/>
![Licence plate for car ID_No 2 detected on2021-11-21 at 06_47_17_212563](https://user-images.githubusercontent.com/93785299/142752907-2a6d3f4a-4e5b-4608-8531-5890ee8ca799.png)
<br />

## Saftey measures:
After Vehicle's classification, traffic frequency and vehicle tracking, it will flag the vehicles and give the output in the csv format based on the provided details. To run, use the following commands:

>(base) Project_folder>**python main.py "Video+Licence plate+Text"** # To run on **VGG16** CNN architecture  <br />
**or**<br/>
>(base) Project_folder>**python main.py "Video+Licence plate+Text" "VGG"** # To run on **VGG16** CNN architecture  <br />
**or**<br/>
>(base) Project_folder>**python main.py "Video+Licence plate+Text" "mobile"** # To run on **VGG16** CNN architecture for vehicle's classification, traffic frequency and safety measure and **MobileNetV2** for vehicle tracking <br />

**Demo video**

https://user-images.githubusercontent.com/93785299/142752504-050ae8f9-c1a6-4fce-88da-37526b69a164.mp4

<br />

**Output images**
 <br />
**1. Vehicle Image**
<br/>
![car ID_No 2 is going in direction up on 2021-11-21 at 06_17_42_816463](https://user-images.githubusercontent.com/93785299/142752869-49aa7d68-78c7-409f-bc47-cc17c0113d4b.png)
<br />
**2. Licence plate image**
<br/>
![Licence plate for car ID_No 2 detected on2021-11-21 at 06_47_17_212563](https://user-images.githubusercontent.com/93785299/142752907-2a6d3f4a-4e5b-4608-8531-5890ee8ca799.png)
<br />
**3. Licence plate text**
<br/>
![Licence plate for car ID_No 2 detected on2021-11-21 at 07_16_35_185054](https://user-images.githubusercontent.com/93785299/142753307-c900ab54-ed3f-4843-a2e4-463987fd1636.png)

<br />
