#Behavioral Cloning Project

**Content**
* Model Architecture Design
* Data Preprocessing
* Model Training

[//]: # (Image References)

[image1]: ./model.png "Model design"
[image2]: ./data-set.png "Model design"

###Model Architecture Design
For the model architecture I chose the comma.ai network that showed good initial result. After running some tests and reading udacity forum I added a cropping layer, that crops top and bottom 25 pixes of an image that don't lead to loose of useful information but reduces the image height from 160 to 110 pixels. 
Each layer is activated by a ELU. I use two dropout layers - after each of fully connected layers - to reduce the possibility of overfitting.

I optimized the model with an Adam optimizer over MSE loss.

The Network design is shown below:
![Undistored image][image1]

I want to point out the MaxPooling layer right after the Cropping layer. Introducing this layer significantly increased training speed without loose in accuracy.

###Data Preprocessing
I used original udacity data set, but as you can see on the image below the angles distribution is not even, I added more turns by flipping images with the turns and adding them to the original data set . Besides that, I collected more data around tricky corners since most of the tracks were straight or soft curves. I drove these corners several times with different speed and steering angle.

![Angles distribution][image2]

I also experimented with different color spaces: grayscale, RGB, HSV and YUV. HSV and YUV showed the best result.

###Model Training
I chose a learning rate of 0.0001 and trained for 10 epochs because performance increase started diminishing when training for longer.

I didn't use a test data set because the real testing could be done by running the simulator in autonomous mode to get qualitative results.
The model shows the best performance at smallest resolution and lowest graphics