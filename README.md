# NMF-For-Classification-of-Defects-on-Steel-Surface-
Non-negative matrix factorization is applied for classification of defects on steel surface using CNN.

For this research the source of Image dataset is been downloaded from the NEU surface defect database. 
And it contains 1800 grayscale images in jpg format belonging to six different classes. The link for the dataset is given below.
http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html.

Two different experiment is performed to find out which techniques gives better results.


First experiment:  refer to code all_with_nmf_final

Step one - The first step in pre-processing of images is to resize
the original images. The actual images contain of 200 x 200 pixels. Which were
resized with the help of cv2.resize().Image width and height were changed by preserving the original aspect ratio. 
Images were resized to 40 x 40.
Step two - After resizing the images, image scaling is done to bring the data to
specifc range. It is performed by dividing the images by 255.
Step three - following the above two steps Gaussian blur and NMF applied then data is divided into training set and test set with 80:20
ratio further this our deep CNN model is trained on 20 epochs..

Results: 45% of accuracy is achieved

Second experiment: refer to code all_without_GB_final

The images were converted to grayscale with the help of rgb2gray() function. NMF is applied and
normalization is done then data is divided into training set and test set with 80:20
ratio further this our deep CNN model is trained on 20 epochs.

Results: 93% of accuracy is achieved.


