import os
import glob
import cv2
import numpy as np

INPUT_FOLDER = "pictures"
OUTPUT_FOLDER = "aligned"


def find_sun_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(biggest)
    return center_x, center_y


def move_sun_to_center(image, center_x, center_y, base_center_x, base_center_y):
    height, width = image.shape[:2]
    shift_x = base_center_x - center_x
    shift_y = base_center_y - center_y
    
    matrix = np.float32([[1, 0, shift_x],
                         [0, 1, shift_y]])
    result = cv2.warpAffine(image, matrix, (width, height))
    return result


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
paths = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

base_image = cv2.imread(paths[0])
base_center_x, base_center_y = find_sun_center(base_image)
print(f"base:  x={base_center_x:.2f}, y={base_center_y:.2f}")

for path in paths[1:]:
    image = cv2.imread(path)

    center_x, center_y = find_sun_center(image)
    result = move_sun_to_center(image, center_x, center_y, base_center_x, base_center_y)

    #Check the new center after moving
    new_x, new_y = find_sun_center(result)
    print(f"было ({center_x:.2f}, {center_y:.2f})  ->  стало ({new_x:.2f}, {new_y:.2f})")



    file_name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, file_name), result)
    print(file_name, "готово")