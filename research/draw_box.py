import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("data/raw/rdd2022/India/train")
NAME = "India_000005"


def read_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bb = obj.find("bndbox")
        x1 = int(float(bb.find("xmin").text))
        y1 = int(float(bb.find("ymin").text))
        x2 = int(float(bb.find("xmax").text))
        y2 = int(float(bb.find("ymax").text))
        boxes.append((label, x1, y1, x2, y2))
    return boxes


def draw(image_path, boxes, out_path):
    img = cv2.imread(str(image_path))
    for label, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    xml_path = ROOT / "annotations" / "xmls" / f"{NAME}.xml"
    image_path = ROOT / "images" / f"{NAME}.jpg"
    out_path = Path("reports/figures") / f"{NAME}_boxes.jpg"

    boxes = read_boxes(xml_path)
    print("boxes found:", boxes)
    draw(image_path, boxes, out_path)
    print("saved to:", out_path)