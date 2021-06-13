# Object Detection from images (Poker Card)
A project to master object detection skills, using deep learning techniques

## **The Goal**
To be able to detect multiple custom objects within one image and to be able to use the detection results for separate programming applications.
This time we selected Poker card as the custom object and we would output an optimal strategy for the game Blackjack based on the cards detected on the betting table.

## **The Approach**
![image](https://user-images.githubusercontent.com/80243823/121808107-9fe36d80-cc89-11eb-96ba-2988442e85a9.png)



## **Base Model Construction**
There are publicly available object detection models online. However, they are for general objects detection. In order to detect the 52 distinct Poker cards, we have to apply transfer learning from one model and then further train the model to suit our needs. 

Comparison of the performance (average precision = AP) of different models available online is as follows,
![image](https://user-images.githubusercontent.com/80243823/121804693-4889d100-cc7a-11eb-9cf5-dc4e2a755ac0.png)
We chose YOLOv4 as our base model. 

With the base model selected, we would then train the model to recognize the card itself, as well as the location of the card within an image.

 
## **Data collection**
The data required for this project is unique. Not only we have to feed the model the correct card label, we have to feed the model the correct location of the card within an image. We couldn't find one available dataset online. For that, we created our own dataset using Poker card images we found online.

First, we downloaded 165 Poker card images images online. Then we used a community tool to create 52 unique classes and labeled the images one by one using the classes. The tool would then ouput a txt file, containing coordinates and size of each label, as well as the correct identification of the card. 

![image](https://user-images.githubusercontent.com/80243823/121810398-513ad100-cc93-11eb-9915-bc281dd3c8a6.png)


The video below showcases the process in details,

https://user-images.githubusercontent.com/80243823/121808460-319faa80-cc8b-11eb-9288-f41d10a3670a.mp4

 

## **Initial Model Training**
![image](https://user-images.githubusercontent.com/80243823/121809713-74b04c80-cc90-11eb-92f2-e5bac5770643.png)
![image](https://user-images.githubusercontent.com/80243823/121809837-07e98200-cc91-11eb-8a6e-dda3b8db2aa9.png)


## **Upsampling**
As the dataset for training was not sufficient, we downloaded and labeled more Poker card images online. Not only that, we applied image augmentation to introduce variation to the dataset we had.

![image](https://user-images.githubusercontent.com/80243823/121810618-2ac96580-cc94-11eb-88d9-f55e1840fc5d.png)


## **New Model and Perforamnce Comparison**
![image](https://user-images.githubusercontent.com/80243823/121810826-decaf080-cc94-11eb-8a6c-a55f3fc2ce23.png)
![image](https://user-images.githubusercontent.com/80243823/121811097-d3c49000-cc95-11eb-9625-00ddd6fefe16.png)
![image](https://user-images.githubusercontent.com/80243823/121811136-fe164d80-cc95-11eb-8016-801187a0ba7c.png)
![image](https://user-images.githubusercontent.com/80243823/121811156-14240e00-cc96-11eb-89f0-d151170259dd.png)
![image](https://user-images.githubusercontent.com/80243823/121811206-4897ca00-cc96-11eb-9d74-931c8307cef2.png)
![image](https://user-images.githubusercontent.com/80243823/121811271-71b85a80-cc96-11eb-9b00-3ac87ad7ecab.png)
![image](https://user-images.githubusercontent.com/80243823/121811286-7ed54980-cc96-11eb-8e91-1379ddccb5c4.png)
![image](https://user-images.githubusercontent.com/80243823/121811305-9280b000-cc96-11eb-9e91-b2ec9623e563.png)
![image](https://user-images.githubusercontent.com/80243823/121811319-9c0a1800-cc96-11eb-9529-638cb80782c4.png)
![image](https://user-images.githubusercontent.com/80243823/121811333-b0e6ab80-cc96-11eb-9072-234bbcd33709.png)


## **Challenges Encountered**
There are 3 major challenges we encountered during the development of the model,

1. Insufficient dataset

Although Model 2 has 1100 images to train with, it is far from being sufficient. Due to the unique requirements of the dataset, we couldn't find available resources online but to create our own dataset, which consumed lots of manhour.


2. Long training time

The neural network we employed was complex, with 100+ hidden layers for image feature extraction (CNN network) alone, each with 100+ nodes. Besides, the model has to distinguish 52 classes (52 unique Poker cards). Under such level of complexity, it took more than 3 full days of training to reach satisfactory level of accuracy, while under constant tuning of hyperparameters of the learning rate.


3. Hyperparameters tuning

The most impactful factor for the training time and model accuracy is the learning rate of the model.
