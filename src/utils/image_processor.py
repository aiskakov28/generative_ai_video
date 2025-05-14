import cv2

class ImageProcessor:
    @staticmethod
    def load_image(path):
        return cv2.imread(path)

    @staticmethod
    def resize_image(image, size=(224, 224)):
        return cv2.resize(image, size)
