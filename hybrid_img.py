# Computer Vision Laboratory by Reah Mae Sardañas



import numpy as np
import cv2

# ------MAIN FUNCTION------------------------------------------------------------------------------------
def main():
    # Load images
    img1 = cv2.imread("input/left.png")
    img2 = cv2.imread("input/right.png")

    
    print("\nENTER THE NEEDED VALUES (separated by commas)")

    # Get user input for sigma, size, and frequency used as a single array separated by comma for the first Image
    first_pic = [float(x) if i == 0 
                    else x.strip() for i,
                         x in enumerate(input("Sigma, Size, and Frequency(low/high) of the 1ST IMAGE: ").split(','))]

    # Assign the values to sigma1, size1, and high_low1
    sigma1 = first_pic[0]
    size1 = int(first_pic[1])
    high_low1 = first_pic[2].lower()

    # Get user input for sigma and size as a single array separated by comma for the second Image
    second_pic = [float(x) if i == 0 
                    else x.strip() for i,
                         x in enumerate(input("Sigma, Size, and Frequency(low/high) of the 2ND IMAGE: ").split(','))]

    # Assign the values to sigma2, size2, and high_low2
    sigma2 = second_pic[0]
    size2 = int(second_pic[1])
    high_low2 = second_pic[2].lower()
    
    # Ask the user if they want to grayscale the images
    grayscale_option = input("\nDo you want to grayscale the images? (yes/no): ").lower()
    if grayscale_option == "yes":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    mixin_ratio = 0.5
    scale_factor = 1.0  # Scale factor to map the image back to [0, 255]

    # update the text file with the new values from the user
    update_text_file(sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio)

    # Create hybrid image
    hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                                     high_low2, mixin_ratio, scale_factor)

    print("Hybrid processing DONE!")

    # Display or save your hybrid image here
    cv2.imshow('Hybrid Image', hybrid_img)
    # Save the hybrid image
    cv2.imwrite("output/hybrid.png", hybrid_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def update_text_file(sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio):
    # Read the existing content of the text file
    with open('hybrid_img_README.txt', 'r') as file:
        lines = file.readlines()

    # Find and update the lines containing the parameters you want to change
    for i, line in enumerate(lines):
        if "Sigma for Image 1" in line:
            lines[i] = f"   Sigma for Image 1: {sigma1}\n"
        elif "Size for Image 1" in line:
            lines[i] = f"   Size for Image 1: {size1}\n"
        elif "Used frequencies from Image 1" in line:
            if high_low1 == 'low':
                lines[i] = "   Used frequencies from Image 1: low\n"
            else:
                lines[i] = "   Used frequencies from Image 1: high\n"

        elif "Sigma for Image 2" in line:
            lines[i] = f"   Sigma for Image 2: {sigma2}\n"
        elif "Size for Image 2" in line:
            lines[i] = f"   Size for Image 2: {size2}\n"
        elif "Used frequencies from Image 2" in line:
            if high_low2 == 'low':
                lines[i] = "   Used frequencies from Image 2: low\n"
            else:
                lines[i] = "   Used frequencies from Image 2: high\n"

    # Write the updated content back to the text file
    with open('hybrid_img_README.txt', 'w') as file:

        file.writelines(lines)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio, scale_factor):
    print("\nProcessing Images...")

    # Preprocess images if necessary and convert to float32 (if the image was converted to grayscale)
    if isinstance(img1, np.ndarray) and img1.dtype == np.uint8:  # checks if image1 is a numpy array and an unsigned 8 bit integer
        img1 = img1.astype(np.float32) / 255.0          #normalizing the values to ensure that the pixel values are within 0 to 1           
        img2 = img2.astype(np.float32) / 255.0


    # Apply filters according to user specifications
    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    
    
    # Blend images using the mixing ratio
    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio

    cv2.imwrite("output/left_output.png", img1 *255)
    cv2.imwrite("output/right_output.png", img2 * 255)

    hybrid_img = (img1 + img2) * scale_factor 

    # Clip and convert the image to uint8 if necessary to handle grayscale image
    if isinstance(hybrid_img, np.ndarray) and hybrid_img.dtype == np.float32:
        hybrid_img = (hybrid_img * 255).clip(0, 255).astype(np.uint8)    

    return hybrid_img

def low_pass(img, sigma, size):
    # Generate Gaussian blur kernel
    kernel = gaussian_blur(sigma, size)
    
    # Perform cross-correlation of the image with the Gaussian blur kernel
    low_pass_img = convolution(img, kernel)

    return low_pass_img

    

def high_pass(img, sigma, size):
    # Get the low-pass filtered image
    low_passed_img = low_pass(img, sigma, size)
    
    # Compute the high-pass filtered image
    high_pass_img = img - low_passed_img
    
    return high_pass_img

def gaussian_blur(sigma, size):
    # Create Gaussian kernel
    center = size // 2  #reference point for gaussian blur
    kernel = np.zeros((size, size))

    # Compute the Gaussian blur kernel by iterating through the row and column indices
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
        
    kernel /= np.sum(kernel)  # Normalize the kernel ro ensure that the sum is equal to 1 (to maintain brightness in convolution)

    return kernel # returns generated blur kernel

def convolution(img, kernel):
    
    # Ensure kernel dimensions (specified SIZE) are odd
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    # Use the cross_correlation function to perform convolution
    return cross_correlation(img, np.flipud(np.fliplr(kernel)))   #flips the kernel both horizontally and vertically 

def cross_correlation(img, kernel):
    
    # Check if the image is grayscale or color and get its dimensions and kernel
    if len(img.shape) == 2:  # Grayscale image
        img_height, img_width = img.shape
        channels = 1
    elif len(img.shape) == 3:  # Color image
        img_height, img_width, channels = img.shape
    else:
        raise ValueError("Invalid image shape.")


    kernel_height, kernel_width = kernel.shape

    # Padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2


     # Initialize an empty output image
    output = np.zeros_like(img)

    if channels == 1:  # Grayscale image
        # Add padding to the image
        img_padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Iterate over each pixel in the image
        for y in range(img_height):
            for x in range(img_width):
                # Perform cross-correlation
                output[y, x] = np.sum(img_padded[y:y + kernel_height, x:x + kernel_width] * kernel)

    elif channels == 3:  # Color image
        # Add padding to each channel of the image
        img_padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

        # Iterate over each pixel in the image
        for y in range(img_height):
            for x in range(img_width):
                # Perform cross-correlation separately for each channel
                for c in range(channels):
                    output[y, x, c] = np.sum(img_padded[y:y + kernel_height, x:x + kernel_width, c] * kernel)

    return output


if __name__ == "__main__":
    main()
