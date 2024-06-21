from PIL import Image
from helper import *
import streamlit as st


def main():
    st.title("CCTV Image enhancement & object detection")
    enhancement_option = st.sidebar.multiselect(
        "Select Enhancement Technique",
        ("Contrast Stretching", "Histogram Equalization", "median Blur", "FFT-highpass-filter", "Sharpening", "Homomorphic Filtering", "Object Detection")
    )
    
    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    print(enhancement_option)
    if image_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enhanced_image = image
        if "Contrast Stretching" in enhancement_option:
            enhanced_image = cont_stretch(enhanced_image)
        if "Histogram Equalization" in enhancement_option:
            enhanced_image = histogramEq(enhanced_image)
        if  "median Blur" in   enhancement_option:
            enhanced_image = medianBlur(enhanced_image)
        if  "FFT-highpass-filter" in   enhancement_option:
            cutoff = st.slider("Cutoff frequency", min_value=10, max_value= 400)
            enhanced_image = highpass_filter_fft(enhanced_image, cutoff=cutoff)
        if  "Sharpening"  in enhancement_option:
            enhanced_image = sharpening(enhanced_image)   
        if "Homomorphic Filtering"  in enhancement_option:
            enhanced_image = HomoMorphicFlitering(enhanced_image) 
        if "Object Detection" in enhancement_option:
            enhanced_image = yolo(enhanced_image)
          
        st.write("Actual Image")
        st.image(image)   
        st.write("Enhanced Image")    
        st.image(Image.fromarray(enhanced_image), use_column_width=True)

if __name__ == "__main__":
    main()

        