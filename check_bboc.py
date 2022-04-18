import cv2
import pandas as pd


file_annot = pd.read_csv("annotation.csv")
file_names = file_annot['filename']
file_x = file_annot['x']
file_y = file_annot['y']
file_width = file_annot['width']
file_height = file_annot['height']
file_classes = file_annot['class']

idx_img = 0

for file in file_names:
    # print("NAME ", fold + "/" + name)
    class_name = file_classes[idx_img]
    full_name = "frames_labeled/valid_faces/" + class_name + "/" + file
    img = cv2.imread(full_name)
    xmin = file_x[idx_img]
    ymin = file_y[idx_img]
    xmax = int(xmin) + int(file_width[idx_img])
    ymax = int(ymin) + int(file_height[idx_img])

    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)

    cv2.imshow("windows_sucks", img)
    s = cv2.waitKey(0)
    if s == ord("q"):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    idx_img += 1
