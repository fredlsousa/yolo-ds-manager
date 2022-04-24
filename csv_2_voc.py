import pandas as pd
from yattag import Doc, indent
import cv2
from tqdm import tqdm
import shutil


ds_annot = pd.read_csv("annotation_correct.csv")

name_list = ds_annot['filename']
x_list = ds_annot['x']
y_list = ds_annot['y']
width_list = ds_annot['width']
height_list = ds_annot['height']
class_list = ds_annot['class']

train_file = open("data/train.txt", "w")
test_file = open("data/test.txt", "w")

# j = 11000
ctrl = 1
with tqdm(total=len(name_list)) as pbar:
    for i in range(len(name_list)):
        doc, tag, text = Doc().tagtext()
        # img = cv2.imread("dataset/side_face/dataset_1/norm_imgs/" + str(name_list[i]))
        img = cv2.imread("./frames_labeled/Lateral_face_filtered/" + class_list[i].strip("\n") + "/" + name_list[i])
        if img is None:
            print("Img == None")
            pass
        else:
            height, width, depth = img.shape
            with tag('annotation'):
                with tag('folder'):
                    if class_list[i].strip("\n") == "lateral_direita":
                        text("lateral_direita")
                    elif class_list[i].strip("\n") == "lateral_esquerda":
                        text("lateral_esquerda")
                with tag('filename'):
                    text(str(name_list[i]))
                with tag('path'):
                    if class_list[i].strip("\n") == "lateral_direita":
                        text("lateral_direita/" + name_list[i])
                    elif class_list[i].strip("\n") == "lateral_esquerda":
                        text("lateral_esquerda/" + name_list[i])
                with tag('source'):
                    with tag('database'):
                        text('Unknown')
                with tag('size'):
                    with tag('width'):
                        text(str(width))
                    with tag('height'):
                        text(str(height))
                    with tag('depth'):
                        text(str(depth))
                with tag('segmented'):
                    text(str(0))
                with tag('object'):
                    with tag('name'):
                        if class_list[i].strip("\n") == "lateral_direita":
                            text("lateral_direita")
                        elif class_list[i].strip("\n") == "lateral_esquerda":
                            text("lateral_esquerda")
                    with tag('pose'):
                        text('Unspecified')
                    with tag('truncated'):
                        text(str(0))
                    with tag('difficult'):
                        text(str(0))
                    with tag('bndbox'):
                        with tag('xmin'):
                            text(str(x_list[i]))
                        with tag('ymin'):
                            text(str(y_list[i]))
                        with tag('xmax'):
                            text(str(width_list[i] + x_list[i]))
                        with tag('ymax'):
                            text(str(height_list[i] + y_list[i]))

            result = indent(doc.getvalue(), indentation=' ' * 4, newline='\r\n')
            # annot_file_name = " "
            annot_file_name = "Annotations/" + str(name_list[i]).split(".")[0] + ".xml"

            # main_file.write(str(name_list[i]).split("/")[1].split(".")[0] + "\n")
            # annot_file_name = "dataset/side_face/dataset_1/Annotations/" + str(name_list[i]) + ".xml"
            # main_file.write(str(name_list[i]) + "\n")
            # main_file.write(str(name_list[i]).split("/")[1].split(".")[0] + "\n")
            # main_file.write(str(name_list[i]).split("/")[1]. + "\n")
            if ctrl <= 8:
                train_file.write("data/left_rigth_ds/" + str(name_list[i]) + "\n")
            else:
                test_file.write("data/left_right_ds/" + str(name_list[i]) + "\n")
                ctrl = 0
            with open(annot_file_name, "w") as f:
                f.write(result)
            # shutil.copyfile("./frames_labeled/Lateral_face_filtered/" + class_list[i].strip("\n") + "/" + name_list[i],
            #                "data/left_right_ds/" + name_list[i])
            # j += 1
            ctrl += 1

        pbar.update(1)

train_file.close()
