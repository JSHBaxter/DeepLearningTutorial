import cv2
import scipy.ndimage as spi

pi = 3.141592

def generate_image(input_img, rotation, img_size):
    output_img = cv2.resize(input_img, dsize=(img_size,img_size))
    output_img = spi.rotate(output_img, 180 * rotation / pi, reshape=False)
    return output_img