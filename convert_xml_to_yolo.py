import os
import xml.etree.ElementTree as ET

# paths
xml_folder = "archive/annotations"
image_folder = "archive/images"
label_folder = "archive/dataset/labels/train"

os.makedirs(label_folder, exist_ok=True)

classes = ["With Helmet", "Without Helmet"]

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_file = os.path.join(label_folder, xml_file.replace(".xml", ".txt"))

    with open(txt_file, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text

            if cls not in classes:
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            ymin = float(xmlbox.find("ymin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymax = float(xmlbox.find("ymax").text)

            # FIX: normalization (VERY IMPORTANT)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h

            # clamp values (extra safety)
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            box_width = min(max(box_width, 0), 1)
            box_height = min(max(box_height, 0), 1)

            f.write(f"{cls_id} {x_center} {y_center} {box_width} {box_height}\n")

print("✅ FIXED conversion done")