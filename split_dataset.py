import os
import random
import xml.etree.ElementTree as ET

master_annotation_file = "plate_dataset/annotations.xml"
source_image_dir = "plate_dataset/photos"
output_dir = "dataset"
train_ratio = 0.7  # 70% trening, 30% test

if not os.path.exists(master_annotation_file):
    print(f"BŁĄD: Nie znaleziono pliku: '{master_annotation_file}'")
    exit()

tree = ET.parse(master_annotation_file)
root = tree.getroot()

file_pairs = []
# Znajdź wszystkie tagi 'image' w pliku, aby przetworzyć każdą adnotację.
all_image_annotations = root.findall('.//image')

for image_element in all_image_annotations:
    image_name = image_element.get('name')
    if image_name:
        image_path = os.path.join(source_image_dir, image_name)
        if os.path.exists(image_path):
            file_pairs.append((image_element, image_path))
        else:
            print(f"Ostrzeżenie: Nie znaleziono pasującego pliku obrazu dla '{image_name}'")

random.shuffle(file_pairs)
split_index = int(len(file_pairs) * train_ratio)
train_files = file_pairs[:split_index]
test_files = file_pairs[split_index:]

train_images_path = os.path.join(output_dir, "images", "train")
test_images_path = os.path.join(output_dir, "images", "test")
train_annotations_path = os.path.join(output_dir, "annotations", "train")
test_annotations_path = os.path.join(output_dir, "annotations", "test")

for path in [train_images_path, test_images_path, train_annotations_path, test_annotations_path]:
    os.makedirs(path, exist_ok=True)

def process_fileset(files, dest_annotations_path, dest_images_path):
    for image_element, source_image_path in files:
        # Przenieś plik obrazu
        image_filename = os.path.basename(source_image_path)
        os.rename(source_image_path, os.path.join(dest_images_path, image_filename))

        # Utwórz nowy, osobny plik .xml dla tej jednej adnotacji
        annotation_filename = os.path.splitext(image_filename)[0] + ".xml"
        destination_xml_path = os.path.join(dest_annotations_path, annotation_filename)
        new_tree = ET.ElementTree(image_element)
        new_tree.write(destination_xml_path, encoding='utf-8', xml_declaration=True)

process_fileset(train_files, train_annotations_path, train_images_path)
process_fileset(test_files, test_annotations_path, test_images_path)

print("\nZakończono!")