import cv2
import numpy as np
from ultralytics import YOLO

def medianBlur(img, kernel_size=3):
    img = cv2.medianBlur(img, kernel_size)
    return img

def highpass_filter_fft(image, cutoff):
    # Convert image to float32
    img_float32 = np.float32(image)

    # Split the image into its RGB channels
    r, g, b = cv2.split(img_float32)

    # Apply FFT to each channel
    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)

    # Shift zero frequency component to center
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)

    # Get the shape of the image
    rows, cols = image.shape[:2]

    # Generate a highpass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[int(rows/2)-cutoff:int(rows/2)+cutoff, int(cols/2)-cutoff:int(cols/2)+cutoff] = 1

    # Apply the mask to each channel
    fft_r_shifted_filtered = fft_r_shifted * mask
    fft_g_shifted_filtered = fft_g_shifted * mask
    fft_b_shifted_filtered = fft_b_shifted * mask

    # Inverse FFT
    img_r = np.fft.ifftshift(fft_r_shifted_filtered)
    img_g = np.fft.ifftshift(fft_g_shifted_filtered)
    img_b = np.fft.ifftshift(fft_b_shifted_filtered)

    img_r = np.fft.ifft2(img_r)
    img_g = np.fft.ifft2(img_g)
    img_b = np.fft.ifft2(img_b)

    img_r = np.abs(img_r)
    img_g = np.abs(img_g)
    img_b = np.abs(img_b)

    # Merge the filtered channels
    filtered_image = cv2.merge((img_r, img_g, img_b))

    # Normalize the image
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = np.uint8(filtered_image)

    return filtered_image


def cont_stretch(img):

    min_val = np.min(img)
    max_val = np.max(img)

    stretched = ((img - min_val) / (max_val - min_val)) * 255

    stretched = stretched.astype(np.uint8)

    return stretched

def histogramEq(image):
    
    r, g, b = cv2.split(image)

    eq_r = cv2.equalizeHist(r)
    eq_g = cv2.equalizeHist(g)
    eq_b = cv2.equalizeHist(b)

    equalized_image = cv2.merge((eq_r, eq_g, eq_b))

    return equalized_image

def sharpening(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image

def HomoMorphicFlitering(img, cutoff = 75):
    # Convert image to float32
    img_float32 = np.log(np.float32(img) + 1)

    # Split the image into its RGB channels
    r, g, b = cv2.split(img_float32)

    # Apply FFT to each channel
    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)

    # Shift zero frequency component to center
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)

    # Get the shape of the image
    rows, cols = img.shape[:2]

    # Generate a highpass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[int(rows/2)-cutoff:int(rows/2)+cutoff, int(cols/2)-cutoff:int(cols/2)+cutoff] = 1

    # Apply the mask to each channel
    fft_r_shifted_filtered = fft_r_shifted * mask
    fft_g_shifted_filtered = fft_g_shifted * mask
    fft_b_shifted_filtered = fft_b_shifted * mask

    # Inverse FFT
    img_r = np.fft.ifftshift(fft_r_shifted_filtered)
    img_g = np.fft.ifftshift(fft_g_shifted_filtered)
    img_b = np.fft.ifftshift(fft_b_shifted_filtered)

    img_r = np.fft.ifft2(img_r)
    img_g = np.fft.ifft2(img_g)
    img_b = np.fft.ifft2(img_b)

    img_r = np.exp(np.abs(img_r))
    img_g = np.exp(np.abs(img_g))
    img_b = np.exp(np.abs(img_b))

    # Merge the filtered channels
    filtered_image = cv2.merge((img_r, img_g, img_b))

    # Normalize the image
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = np.uint8(filtered_image)

    return filtered_image

def yolo(img):

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    image = img.copy()
    # Perform object detection
    results = model.predict(source=image, save=False, stream=True)

    # Iterate over the detected objects
    for result in results:
        boxes = result.boxes.data.cpu().numpy()  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.astype(int)  # Unpack the box coordinates and confidence
            label = model.names[int(cls)]  # Get the class label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)  # Write the label and confidence

    return image