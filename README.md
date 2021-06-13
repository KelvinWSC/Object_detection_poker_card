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




https://user-images.githubusercontent.com/80243823/121808460-319faa80-cc8b-11eb-9288-f41d10a3670a.mp4

