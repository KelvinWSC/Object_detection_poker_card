# Object Detection from images (Poker Card)
A project to master object detection skills, using deep learning techniques

## **The Goal**
To be able to detect multiple custom objects within one image and to be able to use the detection results for separate programming applications.
This time we selected Poker card as the custom object and we would output an optimal strategy for the game Blackjack based on the cards detected on the betting table.

## **The Approach**
There are publicly available object detection models online. However, they are for general objects detection. In order to detect the 52 distinct Poker cards, we have to apply transfer learning from one model and then further train the model to suit our needs. 

Comparison of the performance (average precision = AP) of different models available online is as follows,
![image](https://user-images.githubusercontent.com/80243823/121804693-4889d100-cc7a-11eb-9cf5-dc4e2a755ac0.png)
We chose YOLOv4 as our base model. 
