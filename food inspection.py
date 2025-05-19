import cv2
import numpy as np

def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    image = cv2.resize(image, (640, 480))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    
    lower_A = np.array([20, 150, 150])
    upper_A = np.array([30, 255, 255])

    
    lower_B = np.array([15, 80, 80])
    upper_B = np.array([25, 180, 180])

    
    lower_C = np.array([0, 50, 0])
    upper_C = np.array([20, 255, 100])

    
    mask_A = cv2.inRange(hsv, lower_A, upper_A)
    mask_B = cv2.inRange(hsv, lower_B, upper_B)
    mask_C = cv2.inRange(hsv, lower_C, upper_C)

    
    count_A = cv2.countNonZero(mask_A)
    count_B = cv2.countNonZero(mask_B)
    count_C = cv2.countNonZero(mask_C)

    
    max_count = max(count_A, count_B, count_C)
    if max_count == count_A:
        quality = 'A - Premium'
        color = (0, 255, 0)
    elif max_count == count_B:
        quality = 'B - Acceptable'
        color = (0, 255, 255)
    else:
        quality = 'C - Reject'
        color = (0, 0, 255)

    
    cv2.putText(image, f'Quality: {quality}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Food Quality", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


classify_image("rotten2.jpg")
