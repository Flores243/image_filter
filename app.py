import cv2 as cv
import gradio as gr
import numpy as np
from PIL import Image

# Filtre fonksiyonları
def gray_filter(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def blur_filter(image):
    return cv.GaussianBlur(image, (15, 15), 0)

def edge_detection(image):
    return cv.Canny(image, 100, 200)

def sepia_filter(image):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    return cv.transform(image, kernel)

def negative_filter(image):
    return cv.bitwise_not(image)

def sharpen_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv.filter2D(image, -1, kernel)

def emboss_filter(image):
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [0,  1, 2]])
    return cv.filter2D(image, -1, kernel)

# Yeni Efektler
def invert_color(image):
    return cv.bitwise_not(image)

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.array(image, dtype=float)
    noisy_image += gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def vignette_filter(image):
    rows, cols = image.shape[:2]
    X_result = cv.getGaussianKernel(cols, cols/5)
    Y_result = cv.getGaussianKernel(rows, rows/5)
    result = Y_result * X_result.T
    mask = result / result.max()
    vignette = np.copy(image)
    for i in range(3):  # Apply to each color channel
        vignette[..., i] = image[..., i] * mask
    return vignette

def contrast_adjustment(image, alpha=2.0, beta=0):
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

# Ana efekt fonksiyonu
def apply_filter(image, filter_type):
    image = np.array(image)
    
    if filter_type == "Gray":
        return gray_filter(image)
    elif filter_type == "Blur":
        return blur_filter(image)
    elif filter_type == "Edge Detection":
        return edge_detection(image)
    elif filter_type == "Sepia":
        return sepia_filter(image)
    elif filter_type == "Negative":
        return negative_filter(image)
    elif filter_type == "Sharpen":
        return sharpen_filter(image)
    elif filter_type == "Emboss":
        return emboss_filter(image)
    elif filter_type == "Invert Color":
        return invert_color(image)
    elif filter_type == "Noise":
        return add_noise(image)
    elif filter_type == "Vignette":
        return vignette_filter(image)
    elif filter_type == "Contrast":
        return contrast_adjustment(image)

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("## Fotoğraf Üzerine Filtre Uygulama")
    gr.Markdown("Bir fotoğraf yükleyin ve istediğiniz efekti seçin.")
    
    # Fotoğraf yükleme alanı
    image_input = gr.Image(type="pil", label="Upload Image")
    
    # Efekt seçimi için dropdown
    filter_choices = [
        "Gray", "Blur", "Edge Detection", "Sepia", "Negative", 
        "Sharpen", "Emboss", "Invert Color", "Noise", "Vignette", "Contrast"
    ]
    filter_dropdown = gr.Dropdown(choices=filter_choices, label="Choose filter", value="Gray")
    
    # Çıktı görüntüsü
    image_output = gr.Image()

    # Efekt uygulama butonu
    btn = gr.Button("Apply Filter")
    btn.click(apply_filter, inputs=[image_input, filter_dropdown], outputs=image_output)

if __name__ == "__main__":
    demo.launch()
