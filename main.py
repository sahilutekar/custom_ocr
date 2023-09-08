# Import the necessary packages
import cv2

from Recq import LicensePlateRecognizer


# Specify the path to the pre-trained model file
model_path = "C:/Users/sahil/Desktop/lpr/alpr/trained_mode/trained_model.h5"

# Specify the target width and height for the cropped images
target_width = 128
target_height = 128

# Create an instance of the LicensePlateRecognizer class
recognizer = LicensePlateRecognizer(model_path, target_width, target_height)

# Specify the path to the image file you want to recognize the license plate from
image_path = "C:/Users/sahil/Desktop/lpr/alpr/alpr1/new_test_ocr/detect/images/MH14HR.png"

# Call the recognize method on the instance and pass the path to the image file
license_plate = recognizer.recognize(image_path)

# Print the recognized license plate
print("License plate:", license_plate)

# Show the image with bounding boxes and predictions
cv2.imshow("License plate recognition", cv2.imread(image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()
