import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm

dirs = ['data/train.txt', 'data/test.txt']
# dirs = ['test.txt']
classes = ['lateral_direita', 'lateral_esquerda']


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x, y, w, h)


# def convert_annotation(xml_path, txt_path, image_path):
def convert_annotation(xml_path, txt_path):
    # basename = os.path.basename(image_path)
    # basename_no_ext = os.path.splitext(basename)[0]

    # in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    # out_file = open(output_path + basename_no_ext + '.txt', 'w')

    in_file = open(xml_path)
    out_file = open(txt_path, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        #     continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()


cwd = getcwd()

for dir_path in dirs:
    txt_file_name = cwd + '/' + dir_path
    # output_path = full_dir_path +'/yolo/'
    # output_path_annot = "dataset-annot/"
    # output_path_img_list = dir_path

    # if not os.path.exists(output_path):
    #    os.makedirs(output_path)

    loaded_txt_file = open(dir_path, "r")
    imgs_list = loaded_txt_file.readlines()
    # write_img_list_darknet = open(output_path_img_list, "w")

    with tqdm(total=len(imgs_list)) as pbar:
        for img in imgs_list:
            img_path = img.rstrip("\n")
            # print(img_path)
            # write_img_list_darknet.write("data/obj/" + img_path + "\n")
            xml_voc_annotation_path = "Annotations/" + img_path.split("/")[2].split(".")[0] + ".xml"
            # print(xml_voc_annotation_path)
            yolo_darknet_annotation_path = "data/left_right_ds/" + img_path.split("/")[2].split(".")[0] + ".txt"
            convert_annotation(xml_voc_annotation_path, yolo_darknet_annotation_path)
            pbar.update(1)
            # print(img)

    # write_img_list_darknet.close()
    loaded_txt_file.close()

    # image_paths = getImagesInDir(txt_file_name)
    # list_file = open(txt_file_name + '.txt', 'w')

    # for image_path in image_paths:
    #     list_file.write(image_path + '\n')
    #     convert_annotation(txt_file_name, output_path, image_path)
    # list_file.close()

    print("Finished processing: " + dir_path)
