# Histogram-Filters-Blending
This Computer Vision project implements histogram equalization, low-pass and high-pass filter, and laplacian blending of images.It is implemented in Python 2.7 and OpenCV 3.3.0

Histogram equalization improves the contrast in an image, in order to stretch out the intensity range. Images can be filtered with various low-pass filters(LPF), high-pass filters(HPF) etc. Image Smoothing is done by convolving the image with LPF, which helps in removing noises, blurring the images etc. HPF filters helps in finding edges in the images. Laplacian Blending is used to blend/stitch images together.



main.py: Python code for Histogram Equalization, LPF, HPF, Laplacian Blending.



input1.jpg: Given input image


input2.png: Given grayscale input image


input3A.jpg: Given input image 1 for Laplacian Blending



input3B.jpg: Given input image 2 for Laplacian Blending


output1.jpg: Result of Histogram Equalization


output2deconv.jpg: Result of Deconvolution


output2HPF: Result of HPF


output2LPF: Result of LPF


output3: Result of Laplacian Blending


