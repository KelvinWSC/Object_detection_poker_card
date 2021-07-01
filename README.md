# Object Detection from Images (Poker Card)
A project to master object detection skills, using deep learning and model tuning techniques

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

First, we downloaded 165 Poker card images online. Then we used a community tool to create 52 unique classes and labeled the images one by one using the classes. The tool would then ouput a txt file, containing coordinates and size of each label, as well as the correct identification of the card. 

![image](https://user-images.githubusercontent.com/80243823/121810398-513ad100-cc93-11eb-9915-bc281dd3c8a6.png)


The video below showcases the process in details,

https://user-images.githubusercontent.com/80243823/122167747-f7622300-cead-11eb-9e8c-670049ecf0d9.mp4


## **Initial Model Training**
![image](https://user-images.githubusercontent.com/80243823/121809713-74b04c80-cc90-11eb-92f2-e5bac5770643.png)
![image](https://user-images.githubusercontent.com/80243823/121809837-07e98200-cc91-11eb-8a6e-dda3b8db2aa9.png)


## **Upsampling**
As the dataset for training was not sufficient, we downloaded and labeled more Poker card images online. Not only that, we applied image augmentation to introduce variation to the dataset we had.

![image](https://user-images.githubusercontent.com/80243823/121810618-2ac96580-cc94-11eb-88d9-f55e1840fc5d.png)

After upsampling, we increased the dataset sizes from 165 images to 1208 images.

## **New Model and Perforamnce Comparison**
![image](https://user-images.githubusercontent.com/80243823/121810826-decaf080-cc94-11eb-8a6c-a55f3fc2ce23.png)
![image](https://user-images.githubusercontent.com/80243823/121811097-d3c49000-cc95-11eb-9625-00ddd6fefe16.png)

Although Model 2 has lower mean average precision, we further tested and anaylzed their performance based on different scenarios.
There are 2 areas which require attention, the accuracy of card location, and the accuracy of card identification.

![image](https://user-images.githubusercontent.com/80243823/121811136-fe164d80-cc95-11eb-8016-801187a0ba7c.png)
![image](https://user-images.githubusercontent.com/80243823/121811156-14240e00-cc96-11eb-89f0-d151170259dd.png)
For sheared image, the accuracy of card identification are similar for both models. However, Model 1 detected King of Heart at the upper left, near the chips area, which is incorrect. 

![image](https://user-images.githubusercontent.com/80243823/121811206-4897ca00-cc96-11eb-9d74-931c8307cef2.png)
For close-up image, both model could detect the location and identification of the Poker cards fairly well. However, Model 2 wrongly detected the inverted 4 of Club as Ace of Club, due to the similarity of the character 4 and A.

![image](https://user-images.githubusercontent.com/80243823/121811271-71b85a80-cc96-11eb-9b00-3ac87ad7ecab.png)
For clustered cards, both model performed equally well.

![image](https://user-images.githubusercontent.com/80243823/121811286-7ed54980-cc96-11eb-8e91-1379ddccb5c4.png)
![image](https://user-images.githubusercontent.com/80243823/121811305-9280b000-cc96-11eb-9e91-b2ec9623e563.png)
![image](https://user-images.githubusercontent.com/80243823/121811319-9c0a1800-cc96-11eb-9529-638cb80782c4.png)
![image](https://user-images.githubusercontent.com/80243823/121811333-b0e6ab80-cc96-11eb-9072-234bbcd33709.png)

Upon close examination of the results, we discovered that Model 1, which used standard stock images for training, performed better for clear, well-defined Poker cards images; While Model 2, which used augmented images for training, performed better for chaotic, messy images.


## **Blackjack Strategy based on Detection Result**
The detection result can be output in a dictionary-like string as below, which can be parsed easily.

```
[
{
 "frame_id":1, 
 "filename":"data/poker.jpg", 
 "objects": [ 
  {"class_id":17, "name":"KC", "relative_coordinates":{"center_x":0.787017, "center_y":0.571584, "width":0.305575, "height":0.481816}, "confidence":0.983804}, 
  {"class_id":16, "name":"10D", "relative_coordinates":{"center_x":0.207783, "center_y":0.720466, "width":0.221507, "height":0.189736}, "confidence":0.994300}, 
  {"class_id":1, "name":"JS", "relative_coordinates":{"center_x":0.364248, "center_y":0.562933, "width":0.122483, "height":0.663116}, "confidence":0.997848}
 ] 
}
]
```

To distinguish dealer's card and player's card, we used y coordinate of the card location as the guideline. As the image/photo is taken from the player's perspective, the dealer's cards are always within the upper part of the image, while the player's cards are within the lower part.

Sometimes, the same card might be detected multiple times if it is shown fully (ranks and suits are printed twice on the same card). To prevent this, we extract a unique list of card from the detection results, as there must be no duplication for a standard 52-card deck.

The strategy of playing Blackjack is a simple probability problem, which can be resolved easily by simulating all possible outcomes. To increase the speed of the program, we hard-coded the calculation results which the program can refer to, instead of doing the calculation every time.

![image](https://user-images.githubusercontent.com/80243823/121839627-1ecdba00-cd0d-11eb-9d39-c268bff28eaf.png)
![image](https://user-images.githubusercontent.com/80243823/121839663-2c833f80-cd0d-11eb-918b-210cc99d6706.png)


## **Realtime Application using Streamlit**
In order to allow the model to be applied anywhere and anytime, we used Streamlit to allow image upload, realtime model threshold tuning and output display.

https://user-images.githubusercontent.com/80243823/122185897-0f42a280-cec0-11eb-8f6b-d94d4d73420f.mp4


After uploading an imgae, the program could identify Ace of Heart as the dealer's hand, Jack of Diamond and Ten of Diamond as the player's hand.
Then it suggested the player to 'Stand'.


## **Challenges Encountered**
There are 3 major challenges we encountered during the development of the model,

1. Insufficient dataset

Although Model 2 has 1100 images to train with, it is far from being sufficient. Due to the unique requirements of the dataset, we couldn't find available resources online but to create our own dataset, which consumed lots of manhour.


2. Long training time

The neural network we employed was complex, with 100+ hidden layers for image feature extraction (CNN network) alone, each with 100+ nodes. Besides, the model has to distinguish 52 classes (52 unique Poker cards). Under such level of complexity, it took more than 3 full days of training to reach satisfactory level of accuracy, while under constant tuning of hyperparameters of the learning rate.


3. Hyperparameters tuning

The most impactful factor for the training time and model accuracy is the learning rate of the model. At first we used 0.001 as the default learning rate. However, we found that the accuracy bounced a lot, which means the loss function was not converging, so we reduced the learning rate to 0.0001. 

That was a mistake.

After a half-day trial and error, we discovered that although the accuracy bounced a lot, its average was actually increasing slowly over time. Upon discovering the trend, we tuned the hyperparameters to be much more aggressive, with learning rate 0.005, lower decay rate of learning rate and momentum ratio of 0.9+.

## **Insight**
1. Training dataset matters a lot

The model can only detect objects which are similar to the training set well. Even a slight variation would reduce the accuracy by a lot. So it is vital to introduce as many variation as possible to mimic real-life situations.

2. Apply high learning rate for initial model training

Even for transferred learning, the weightings of the model may be far away from the optimal settings. High learning rate can help converge to the minimum of the loss function during backpropagation much faster.

3. Apply low learning rate during final stage of deep learning

Once the weightings of the model are close to the optimal settings, low learning rate can prevent "over-shooting" the minimum and converge closer to it.

4. Decay may cause the learning rate to decrease too rapidly

There are many different decay formulas, which make the learning rate to decrease exponentially, inversely or by step. As the model we selected used inverse decrease formula, not only we had to reduce the decay rate, we had to adjust the learning rate higher manually from time to time, to force the neural network to evolve at a higher rate.

5. Momentum helps breaking through the local minimum of loss function

Momentum is a term which specifies how much proportion of gradient decent of previous iteration to retain. Sometimes there are local minimum of the loss function, which causes a wall that make the accuracy of the model stuck to a level without improvement no matter how much it trains. Momentum helps by letting the gradient decent to "overshoot" these local minimum. We initially used 0.9+ for momentum, then gradually reduced it to 0.1 during the final stage of deep learning.

6. Reducing the no. of classes the model has to predict, by breaking down the problem into smaller problems, would increase the accuracy

We trained the neral network to detect 52 different Poker cards. A better approach is to train 2 separate neral networks, one for the ranks (Ace to King), and one for the 4 suits (clubs diamonds, hearts and spades), then combine their results to detect the actual card. This approach reduce the no. of classes from 52 to 17 (13 ranks + 4 suits), which is much less demanding
