import pytesseract
import cv2
import sys
import imutils
from PIL import Image
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np

img = cv2.imread(sys.argv[1])


def find_sudoku_board(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale_image, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    count = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            count = approx
            break

        if count is None:
            raise Exception("Cannot find sudoku board on image!")
        break

    output = image.copy()
    cv2.drawContours(output, [count], -1, (0, 255, 0), 2)

    puzzle = four_point_transform(image, count.reshape(4, 2))
    warped = four_point_transform(grayscale_image, count.reshape(4, 2))

    bw = cv2.bitwise_not(warped)

    horizontal = bw.copy()
    vertical = bw.copy()

    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    grid_extracted = cv2.add(warped, vertical)
    grid_extracted = cv2.add(grid_extracted, horizontal)

    subImage = grid_extracted[0:int(grid_extracted.shape[0]*0.33*0.33), 0:int(grid_extracted.shape[1]*0.33*0.33)]
    show_image(subImage)

    custom_config = r' -l eng --oem 1 --psm 6  -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789 "'
    extracted = pytesseract.image_to_string(subImage, config=custom_config)

    print(extracted)


def show_image(image):
    cv2.imshow("Result", image)
    cv2.waitKey(0)


find_sudoku_board(img)