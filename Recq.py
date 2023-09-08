import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import functools


class LicensePlateRecognizer:
    def __init__(self, model_path, TARGET_WIDTH, TARGET_HEIGHT):
        self.model_path = model_path
        self.model = None
        self.chars = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.lower = None
        self.upper = None
        self.TARGET_WIDTH = TARGET_WIDTH
        self.TARGET_HEIGHT = TARGET_HEIGHT
        self.vehicle_plate = ""

    def ocr_load_model(self):
        self.model = load_model(self.model_path, compile=False)
        return self.model
    
    def _compare_rects(self, rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

    def recognize(self, image_path):
        # import pdb;pdb.set_trace()

        # Read the image and convert to grayscale
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        print(h,w)
        if  90< h < 150 and 250<w<1000:
            image=image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 1)
            thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 19)
        else:
            print("In this")
            h, w, c = image.shape
            new_h, new_w = 4*h, 4*w
            print("height_new" ,new_h )
            print("new_weight",new_w)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5),1)
            # blurred = cv2.GaussianBlur(gray, (7,7),cv2.BORDER_DEFAULT)

            # _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # C = int(otsu_threshold[0] / 2)
            average_size = np.sum(image) / (new_h * new_w)
            block_size = int(average_size / 20)
            if block_size % 2 == 0:
                block_size += 1
            print("b", block_size)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blurring and thresholding
#         # to reveal the characters on the license plate
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         thresh = cv2.adaptiveThreshold(blurred, 255,
#                                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

#         # Perform connected components analysis on the thresholded images and
#         # initialize the mask to hold only the components we are interested in
        _, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")

        

        # Set lower bound and upper bound criteria for characters
        total_pixels = image.shape[0] * image.shape[1]
        self.lower = total_pixels // 85 # heuristic param, can be fine tuned if necessary
        self.upper = total_pixels // 20  # heuristic param, can be fine tuned if necessary

        # Loop over the unique components
        for (i, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore it
            if label == 0:
                continue

            # Otherwise, construct the label mask to display only connected component
            # for the current label
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # If the number of pixels in the component is between lower bound and upper bound,
            # add it to our mask
            if numPixels > self.lower and numPixels < self.upper:
                mask = cv2.add(mask, labelMask)

        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(self._compare_rects) )

        # Sort the bounding boxes from left to right, top to bottom
        # sort by Y first, and then sort by X if Ys are similar
        def compare(rect1, rect2):
            print(f"rect1: {rect1} and rect2: {rect2}")
            if abs(rect1[1] - rect2[1]) > 10:
                return rect1[1] - rect2[1]
            else:
                return rect1[0] - rect2[0]
        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

        for rect in boundingBoxes:

            # Get the coordinates from the bounding box
            x,y,w,h = rect

            # Crop the character from the mask
            # and apply bitwise_not because in our training data for pre-trained model
            # the characters are black on a white background
            crop = mask[y:y+h, x:x+w]
            crop = cv2.bitwise_not(crop)

            # Get the number of rows and columns for each cropped image
            # and calculate the padding to match the image input of pre-trained model
            rows = crop.shape[0]
            columns = crop.shape[1]
            paddingY = (self.TARGET_HEIGHT - rows) // 2 if rows < self.TARGET_HEIGHT else int(0.17 * rows)
            paddingX = (self.TARGET_WIDTH - columns) // 2 if columns < self.TARGET_WIDTH else int(0.45 * columns)
            
            # Apply padding to make the image fit for neural network model
            crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

            # Convert and resize image
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)     
            crop = cv2.resize(crop, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

            # Prepare data for prediction
            crop = crop.astype("float") / 255.0
            crop = img_to_array(crop)
            crop = np.expand_dims(crop, axis=0)

            # Make prediction
            prob = self.ocr_load_model().predict(crop)[0]
            idx = np.argsort(prob)[-1]
            self.vehicle_plate += self.chars[idx]

            # Show bounding box and prediction on image
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
            cv2.putText(image, self.chars[idx], (x,y+15), 0, 0.8, (0, 0, 255), 2)
        
        return self.vehicle_plate

