##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[carnotcar1]: ./output_images/car_not_car-1.png
[carnotcar2]: ./output_images/car_not_car-2.png
[carhog]: ./output_images/car-HOG.png
[noncarhog]: ./output_images/non-car-HOG.png
[sliding_window]: ./output_images/sliding_windows.png
[heatmap1]: ./output_images/heatmap1.png
[heatmap2]: ./output_images/heatmap2.png
[heatmap3]: ./output_images/heatmap3.png
[labels1]: ./output_images/labels1.png
[labels2]: ./output_images/labels2.png
[labels3]: ./output_images/labels3.png
[video1]: ./project_video_output.mp4.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in `main.py` lines 13-45.

I started by reading in all the `vehicle` and `non-vehicle` images. First I create a list of images and save it as a pickle files for later use.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example 1][carnotcar1]
![Example 2][carnotcar2]

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Car HOG example][carhog]
![Non-car HOG example][noncarhog]

####2. Explain how you settled on your final choice of HOG parameters.

I wanted to try several different options, but since there are too many parameters to try them all, I chose to play with color spaces, channels and spatial features on or off.
For that, I generated features for all possible permutations of the following options and save the features as pickle files (48 sets of features in total). 

```
color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
color_channels = [0, 1, 2, "ALL"]
spatial_feats = [True, False]
```

I then trained SVM, Decision Trees and Naive Bayes with each of the options and compared the scores.

SVM turned out to be not only the fastest one to train but also the most accurate one: HLS, All channels, spatial_feat set to true showed 0.9916 accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is in `main.py` lines 266-309. First, I extracted features of car and non-car images, then I scaled the features using `StandardScaler` and generated labels for the data set.
Before fitting the data to the classifier, I split the data into a training set and a test test using `train_test_split`

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is in `main.py` lines 337-344. I tested several sizes of window. During my tests I found that window size less than 64px bring more noise than information and size more than 96px also don't add much information. 
In my final configuration I use 2 sizes: 64px with 0.8 overlap and 96px with 0.6 overlap.

![Sliding window][sliding_window]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
To reduce number of searching windows I restricted the searching area to capture only the road.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and integrated the heatmap over last 10 frames then thresholded that integrated result to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here are examples of heat maps:

![Heatmap][heatmap1]
![Heatmap][heatmap2]
![Heatmap][heatmap3]

### Here is the output of `scipy.ndimage.measurements.label()` on the heatmaps and applyed threshold
![Labels1][labels1]
![Labels2][labels2]
![Labels3][labels3]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problems are false positives and that the pipeline sometimes misses out cars. 
I think it can be solved by tuning the features used to train classifier and tuning the parameters of the classifier itself. 

Another big problem is performance. It's far bing capable to recognize vehicles in real time. 
Performance can be improved by resizing input video stream and using optimized HOG algorithm.
