import cv2
import pytesseract
import numpy as np
import sys

# Check for command-line argument
if len(sys.argv) < 2:
    print("Usage: python sudoku_reader.py <image_filename>")
    sys.exit(1)

image_path = sys.argv[1]

# Load the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image '{image_path}'")
    sys.exit(1)

# Convert to grayscale and apply adaptive threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours to detect the grid
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Assume largest contour is the Sudoku grid
grid_contour = contours[0]
epsilon = 0.02 * cv2.arcLength(grid_contour, True)
approx = cv2.approxPolyDP(grid_contour, epsilon, True)

# Warp perspective to get a top-down view
def get_warp(img, approx):
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    side = max([
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ])

    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (int(side), int(side)))
    return warp

warped = get_warp(gray, approx)

# Divide into 9x9 grid and apply OCR
cell_size = warped.shape[0] // 9
sudoku_array = []

for i in range(9):
    row = []
    for j in range(9):
        x = j * cell_size
        y = i * cell_size
        cell = warped[y:y+cell_size, x:x+cell_size]
        cell = cv2.resize(cell, (100, 100))
        cell = cv2.GaussianBlur(cell, (5, 5), 0)
        text = pytesseract.image_to_string(cell, config='--psm 10 digits')
        digit = int(text.strip()) if text.strip().isdigit() else 0
        row.append(digit)
    sudoku_array.append(row)

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[3*(row//3) + i//3][3*(col//3) + i%3] == num:
            return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

print("\nExtracted Sudoku Puzzle:")
for row in sudoku_array:
    print(row)

if solve_sudoku(sudoku_array):
    print("\nSolved Sudoku Puzzle:")
    for row in sudoku_array:
        print(row)
else:
    print("\nNo solution found for the extracted puzzle.")

