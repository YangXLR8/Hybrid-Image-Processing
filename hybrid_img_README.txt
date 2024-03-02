
COMPUTER VISION 
LABORATORY 2: IMAGE HYBRID PROCESSING
by Reah Mae Sardañas


Hybrid Filter Parameters:
    Mix-in Ratio: 0.5
    scale_factor: 1.0

   Sigma for Image 1: 5.0
   Size for Image 1: 15
   Used frequencies from Image 1: low
     
   Sigma for Image 2: 15.0
   Size for Image 2: 25
   Used frequencies from Image 2: high

For this laboratory:
  1. User will enter 3 values for (sigma, size, and frequency) for each image.
  2. User has an option to co vert the image to grayscale or not before hybrid processing.
 
However, this program does not have (current version):
  1. Error handling when the user enter an even number for size values at the start.
      * Convolution with an odd-sized kernel ensures that the output image has the same dimensions as the input image.
  2. Error handling when the user enter both low or both high for the frequencies of the images. 
      * low-pass filters are used to blur an image and remove high-frequency noise
      * high-pass filters are used to enhance edges and detect fine details in an image

To run the program:
  python hybrid_img.py

  (Optional. You can put your own values.)
  for image1: 5, 15, low
  for image2: 15, 25, high