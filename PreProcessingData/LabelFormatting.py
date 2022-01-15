import csv
import cv2
import os

annotations_path = "/content/drive/MyDrive/Dataset/train/Person/labels/"
images_path = "/content/drive/MyDrive/Dataset/train/Person/images/"

header = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
with open(annotations_path + 'labels.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(header)
    for textfile in os.listdir(annotations_path):
        image = cv2.imread(images_path + textfile[:-4] + ".jpg")
        if image is None:
            continue
        width = image.shape[1]
        height = image.shape[0]
        with open(annotations_path + textfile, 'r', newline='') as f_text:
            csv_text = csv.reader(f_text, delimiter=':', skipinitialspace=True)
            rows = list(csv_text)
            for row in rows:
                csv_list = [textfile[:-4]]
                values = [float(r[:5]) for r in row[0].split(' ')]
                csv_list.append(max(0, round(values[1] * width - values[3] * width / 2)))
                csv_list.append(min(width, round(values[1] * width + values[3] * width / 2)))
                csv_list.append(max(0, round(values[2] * height - values[4] * height / 2)))
                csv_list.append(min(height, round(values[2] * height + values[4] * height / 2)))
                csv_list.append(1)
                csv_output.writerow(csv_list)
