import cv2
import os
from tqdm import tqdm

# Workflow:
#           1 - Rodar esse script para filtrar faces laterais esquerda e direita
#           2 - Filtrar frames para apenas frames com faces laterais
#           3 - Anotar bounding boxes nas imagens apenas com faces laterais


def save_frames(video_path_name, folder_name):
    cap = cv2.VideoCapture(video_path_name)
    if not cap.isOpened():
        print("Can't open video!")
        exit()
    frame_id = 0
    os.mkdir(folder_name)
    # while True:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("Not valid ret. Exiting!")
            break
        frame = cv2.resize(frame, (416, 416))
        img_name = folder_name + "/" + folder_name.split("/")[-1] + "_" + str(frame_id) + ".jpg"
        cv2.imwrite(img_name, frame)
        frame_id += 1


ids = [1, 2, 3]
os.mkdir("frames_labeled")
esq_folder = "./frames_labeled/lateral_esquerda"
dir_folder = "./frames_labeled/lateral_direita"
os.mkdir(esq_folder)
os.mkdir(dir_folder)
for id in ids:
    video_path = "/home/fred/Documents/anotacao_marco_22/videos4anotar/0" + str(id) + "/"
    label_file = open("label_files/annotation_0" + str(id) + ".txt")
    lines_label = label_file.readlines()
    with tqdm(total=len(lines_label)) as pbar:
        for line in lines_label:
            filename = line.split(";")[0]
            label = line.split(";")[1].strip("\n")
            if label == "lateraldireita":
                video_name = video_path + filename
                folder = filename.split(".")[0]
                save_frames(video_path + filename, dir_folder + "/" + folder)

            elif label == "lateralesquerda":
                video_name = video_path + filename
                folder = filename.split(".")[0]
                save_frames(video_path + filename, esq_folder + "/" + folder)

            pbar.update(1)
