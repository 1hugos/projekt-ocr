import os
import xml.etree.ElementTree as ET

dataset_base_dir = "dataset"
sets = ["train", "test"]

def convert_xml_to_yolo(annotations_dir, images_dir, labels_dir):
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        try:
            # Parsowanie pliku XML
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()

            # Pobranie wymiarów obrazu z tagu <image>
            image_width = int(root.get('width'))
            image_height = int(root.get('height'))
            
            # Znajdź tag <box>, który zawiera współrzędne tablicy
            box_element = root.find('box')
            if box_element is None:
                print(f"Ostrzeżenie: Brak tagu <box> w pliku {xml_file}. Pomijam.")
                continue

            # Pobranie współrzędnych z atrybutów
            xtl = float(box_element.get('xtl'))
            ytl = float(box_element.get('ytl'))
            xbr = float(box_element.get('xbr'))
            ybr = float(box_element.get('ybr'))

            # Konwersja współrzędnych na format YOLO (środek x, środek y, szerokość, wysokość)
            box_width = xbr - xtl
            box_height = ybr - ytl
            x_center = xtl + (box_width / 2)
            y_center = ytl + (box_height / 2)

            # Normalizacja wartości (sprowadzenie do zakresu 0-1)
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = box_width / image_width
            height_norm = box_height / image_height
            
            # Identyfikator klasy, jedna klasa dlatego 0
            class_id = 0
            
            label_filename = os.path.splitext(xml_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)

            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania pliku {xml_file}: {e}")

for s in sets:
    current_annotations_dir = os.path.join(dataset_base_dir, "annotations", s)
    current_images_dir = os.path.join(dataset_base_dir, "images", s)
    current_labels_dir = os.path.join(dataset_base_dir, "labels", s)
    
    os.makedirs(current_labels_dir, exist_ok=True)

    convert_xml_to_yolo(current_annotations_dir, current_images_dir, current_labels_dir)

print("\nZakończono konwersję!")