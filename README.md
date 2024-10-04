# FastSAM_ObjectManipulator
For the best experience, it's recommended to run the provided Jupyter notebook __(Avataar_Assignment_H3.ipynb)__ in Google Colab. This environment ensures seamless connectivity to the necessary libraries and resources, allowing for optimal performance during image processing tasks. You can easily upload your own images, execute the code, and visualize the results without any installation hassles. Additionally, I have included a selection of input images and their corresponding generated output images to illustrate the capabilities of the tool.

The link to Colab notebook: https://colab.research.google.com/drive/1Bo9_O51OYjCrhlRQPa4TORpyaMXS5qVI?usp=sharing


# Overview
This Python script, named `image_segmentation.py` that utilizes the FastSAM model for image segmentation and the CLIP model for image embedding generation. It allows users to specify a scene and a text prompt for segmentation and subsequently adjust the position of the segmented object within the image.

__This README is crafted to explain the features and usage to recruiters at Avataar. Thank you for reviewing my assignment—hope you find it interesting!__

# Features
* __Image Upload:__ Easily select an image from your local filesystem using a GUI prompt.
* __Object Segmentation:__ Utilizes the FastSAM model to identify and create masks for specified objects in the image.
* __Mask Visualization:__ Highlights the segmented object with a red mask for clear visibility.
* __Object Relocation:__ Allows users to specify shifts in the x and y directions to move the segmented object.
* __Seamless Inpainting:__ Effectively removes the original object and fills in the background to ensure a natural appearance.

# Workflow
1. __Run the Program__
  * Execute the script in a Python environment, or run it in a Google Colab notebook.
2. __Select an Image__
  * Upon running, a dialog box will prompt you to choose an image. You can also specify the image path directly.
  * The program will automatically extract the object to be segmented based on user input.
3. __Processing__
  * The script will:
    * Load and preprocess the selected image.
    * Load the __CLIP__ model and generate image embeddings.
    * Use the __FastSAM__ model to segment the specified object based on user input.
    * Create a mask of the segmented object and apply a red overlay for visualization.
    * Prompt the user to enter the desired x and y shifts for relocating the object.
    * Shift the segmented object in the image and seamlessly inpaint the background.
4. __View Results__
  * The final image with the relocated object will be displayed and can be saved as ``final_image_with_subject.png``. Intermediate steps will also be visualized for better understanding.

## Dependencies
* __FastSAM__: For object segmentation.
* __CLIP__: For generating image embeddings.
* __OpenCV__: For image processing and manipulation.
* __NumPy__: For efficient numerical operations on matrices.
* __Matplotlib__: For displaying and plotting images for immediate visualization.
* __Pillow__: For image handling.

# Usage
## Task 1: Segmenting an Object with a Red Mask
```bash
python image_segmentation.py --image ./example.jpg --class laptop --output ./generated.png
```
### Arguments:
`--image`: Path to the input image file.\
`--class`: The object class you want to segment (e.g., "laptop").\
`--output`: Path where the output image with the red mask will be saved.

## Task 2: Shifting the Segmented Object
To shift the segmented object in the image, use:
```bash
python image_segmentation.py --x 80 --y 0
```
### Arguments:
`--x`: Number of pixels to shift horizontally (positive for right, negative for left).\
`--y`: Number of pixels to shift vertically (positive for down, negative for up).

# Notes
  * Ensure that the FastSAM model file (``FastSAM-s.pt``) is in the same directory as the script or adjust the path accordingly.
  * Adjust the parameters according to your needs to achieve the desired results.

# Why You Should Consider Me for a Position at Avataar 
I’m truly excited about the opportunity to join __Avataar__! Developing the __FastSAM_ObjectManipulator__ has deepened my understanding of advanced image processing techniques and machine learning models. This project was a hands-on experience that taught me not only technical skills but also resilience in overcoming challenges, thanks to resources like community forums and mentorship.

I bring a strong blend of creativity and analytical thinking, which I believe aligns well with Avataar’s innovative spirit. I’m passionate about crafting solutions that enhance user experiences, and I’m eager to contribute my skills to your team.

Let’s collaborate to push the boundaries of AI and create impactful solutions together!
