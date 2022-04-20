import cv2
import glob
from tqdm import tqdm

source_path = "./frames_labeled/Lateral_face_filtered/lateral_direita/*.jpg"
destination_path = "./frames_labeled/Lateral_face_filtered/flipped_r/"

with tqdm(total=len(glob.glob(source_path))) as pbar:
    for img_name in glob.glob(source_path):
        img = cv2.imread(img_name)
        flipped_img = cv2.flip(img, 1)
        file_save_name = destination_path + img_name.split("/")[-1].split(".")[0] + "_flipped.jpg"
        cv2.imwrite(file_save_name, flipped_img)
        pbar.update(1)
