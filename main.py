import pytesseract
import cv2
import sys
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import easyocr
import time
from sudoku import Sudoku

img = cv2.imread(sys.argv[1])


def main():
    start_time = time.time()
    board = find_sudoku_board(img)
    board_without_grid = extract_grid(board)
    (numbers, indices) = get_numbers_from_board_pytesseract(board_without_grid)
    solved = solve_sudoku(numbers)
    create_solution_file(solved, indices, board)
    print("--- %s seconds ---" % (time.time() - start_time))
    # get_numbers_from_board_easyocr(board)


def create_solution_file(solved, indices, image):
    font_size = 35
    for elem in indices:
        x = elem[0]
        y = elem[1]
        x1 = int(x * image.shape[0] * 0.33 * 0.33)
        x2 = int((x + 1) * image.shape[0] * 0.33 * 0.33)
        y1 = int(y * image.shape[1] * 0.33 * 0.33)
        y2 = int((y + 1) * image.shape[1] * 0.33 * 0.33)
        sub_image = image[x1:x2, y1:y2]
        cv2.putText(sub_image, str(solved[x][y]), (int(sub_image.shape[0] / 3), int(sub_image.shape[1] / 1.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, (x2 - x1) / font_size, (0, 0, 0), 1,
                    cv2.LINE_AA, False)

    cv2.imwrite("solution.jpg", image)
    # show_image(image)


def solve_sudoku(numbers):
    int_numbers = []
    split = numbers.split('\n')
    for x in range(9):
        int_numbers.append(list(map(int, split[x])))

    puzzle = Sudoku(3, 3, int_numbers)
    solution = puzzle.solve()
    solution.show_full()
    return solution.board


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

    warped = four_point_transform(grayscale_image, count.reshape(4, 2))
    return warped


def extract_grid(image):
    bw = cv2.bitwise_not(image)

    horizontal = bw.copy()
    vertical = bw.copy()

    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    grid_extracted = cv2.add(image, vertical)
    grid_extracted = cv2.add(grid_extracted, horizontal)
    return grid_extracted


def get_numbers_from_board_pytesseract(image):
    result_matrix = np.full((9, 9), -1)
    custom_config = r' -l eng --psm 6 -c tessedit_char_whitelist="0123456789"'
    font_size = 35
    indices = []
    for x in range(9):
        for y in range(9):
            x1 = int(x * image.shape[0] * 0.33 * 0.33)
            x2 = int((x + 1) * image.shape[0] * 0.33 * 0.33)
            y1 = int(y * image.shape[1] * 0.33 * 0.33)
            y2 = int((y + 1) * image.shape[1] * 0.33 * 0.33)
            sub_image = image[x1:x2, y1:y2]
            if np.mean(sub_image) > 254:
                indices.append((x, y))
                result_matrix[x][y] = 0
                cv2.putText(sub_image, '0', (int(sub_image.shape[0] / 3), int(sub_image.shape[1] / 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, (x2 - x1) / font_size, (0, 0, 0), 1,
                            cv2.LINE_AA, True)

    extracted = pytesseract.image_to_string(image, config=custom_config)
    return extracted, indices


def get_numbers_from_board_easyocr(image):
    reader = easyocr.Reader(['ch_sim', 'en'], True)
    result = reader.readtext(image, detail=0)
    print(result)


def show_image(image):
    cv2.imshow("Result", image)
    cv2.waitKey(0)


main()
