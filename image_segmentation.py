# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
from ultralytics import FastSAM

def upload_image():
    from google.colab import files
    uploaded = files.upload()  # This will prompt you to upload an image file
    return next(iter(uploaded))

def apply_red_mask(image, mask):
    mask = mask.astype(bool)
    red_mask = np.zeros_like(image)
    red_mask[mask] = [0, 0, 255]
    output_image = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return output_image

def shift_segmented_object(original_image, combined_mask, x_shift, y_shift):
    shifted_image = np.zeros_like(original_image)
    mask_shifted = np.zeros_like(combined_mask)
    output_image = original_image.copy()

    for y in range(combined_mask.shape[0]):
        for x in range(combined_mask.shape[1]):
            if combined_mask[y, x]:
                new_x = min(max(x + x_shift, 0), original_image.shape[1] - 1)
                new_y = min(max(y + y_shift, 0), original_image.shape[0] - 1)
                shifted_image[new_y, new_x] = original_image[y, x]
                mask_shifted[new_y, new_x] = 1
                output_image[y, x] = [0, 0, 255]

    mask_to_inpaint = combined_mask.astype(np.uint8)
    inpainted_area = cv2.inpaint(output_image, mask_to_inpaint, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    output_image[mask_shifted > 0] = shifted_image[mask_shifted > 0]
    output_image = cv2.addWeighted(output_image, 1, inpainted_area, 0.5, 0)

    return output_image

def main():
    # Load the image
    image_path = upload_image()
    original_image = cv2.imread(image_path)

    if original_image is None:
        raise ValueError("Error loading image. Check the path.")

    orig_height, orig_width = original_image.shape[:2]

    # Convert to PIL for CLIP processing
    original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32")

    # Preprocess and generate image embeddings
    image_input = preprocess(original_image_pil).unsqueeze(0)
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(image_input)

    np.save('image_embeddings.npy', image_embeddings.numpy())
    print("Image embeddings saved as 'image_embeddings.npy'")

    # Load FastSAM model
    model = FastSAM("FastSAM-s.pt")

    # User input for segmentation
    user_prompt = input("Enter the object you want to segment: ")
    results = model(image_path, texts=user_prompt)

    # Create combined mask from segmentation results
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    if isinstance(results, list):
        for result in results:
            if result.masks is not None:
                masks_data = result.masks.data.numpy()

                if masks_data.shape[0] > 0:
                    for mask_array in masks_data:
                        binary_mask = mask_array > 0
                        if binary_mask.shape != (orig_height, orig_width):
                            binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                        else:
                            binary_mask_resized = binary_mask

                        combined_mask |= binary_mask_resized

    # Get user input for shifting
    x_shift = int(input("Enter x shift (positive for right, negative for left): "))
    y_shift = int(input("Enter y shift (positive for down, negative for up): "))

    # Shift the segmented object
    shifted_image = shift_segmented_object(original_image, combined_mask, x_shift, y_shift)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB))
    plt.title('Shifted Object with Original Position Highlighted')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
