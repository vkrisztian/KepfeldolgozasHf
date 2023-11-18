import pytesseract
import cv2
import sys
import imutils

print(sys.argv[1])

img = cv2.imread(sys.argv[1])


def find_sudoku_board(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale_image, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    show_image(thresh)
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
    show_image(output)


def show_image(image):
    cv2.imshow("Result", image)
    cv2.waitKey(0)


find_sudoku_board(img)