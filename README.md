<h1 align="center">Hybrid Image Generation</h1>
<h3 align="center">Laboratory Output #2 for Computer Vision</h3>

<p align="center">
  <img width="700" src="https://github.com/YangXLR8/Hybrid-Image-Processing/blob/main/output/hybrid.png" alt="cli output"/>
</p>

## Description

This Laboratory Output generates a hybrid image by combining two input images (`left.png` and `right.png`) based on user-specified parameters for Gaussian blur and frequency.

## Features

- **Image Loading**: Loads two input images (`left.png` and `right.png`).
- **User Input**: Prompts the user to enter values for sigma (standard deviation), size (kernel size), and frequency (low/high) for both images.
- **Grayscale Option**: Provides an option to convert images to grayscale before processing.
- **Image Processing**: Applies Gaussian blur (low-pass or high-pass) based on user inputs.
- **Image Blending**: Blends the processed images using a specified mixing ratio.
- **Text File Update**: Updates a text file (`hybrid_img_README.txt`) with the user-defined parameters.
- **Output**: Displays the hybrid image and saves it as `hybrid.png` in the `output` directory.

## Project Structure

- `Input/`: Input images
- `Output/`: Output folder.
- `00-README.txt`: Laboratory Instructions
- `hybrid_img.py`: Main script to run the Hybrid Image Generation
- `hybrid_img_README.txt` : txt file for specifif information
- `Sardañas_Lab2.zip`: submitted final laboratory output
- `OlivaTorralb_Hybrid_Siggraph06.pdf` : reference

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy (`numpy`)

Install these dependencies using pip if you haven't already:

```bash
pip install opencv-python numpy
```

## Usage

1. Run the script:
```bash
python hybrid_img.py
```
2. Follow the prompts to enter values for sigma, size, and frequency for both images.
3. Choose whether to grayscale the images.
4. View the generated hybrid image displayed using OpenCV window.
5. Find the processed images in the output directory.

## Results

This laboratory generates the the following outputs:

- `output/left_output.png`: Processed left image after Gaussian blur.
- `output/right_output.png`: Processed right image after Gaussian blur.
- `output/hybrid.png`: Final hybrid image blending both processed images.
