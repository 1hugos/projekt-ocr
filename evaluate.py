import os
import time
import cv2
import numpy as np
import pytesseract
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
import csv

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

CROPPED_PLATES_DIR = "evaluation_results/cropped_plates"
DIAGNOSTIC_CSV_PATH = "evaluation_results/diagnostic_report.csv"


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea != 0 else 0


def post_process_text(text: str) -> str:
    if not text:
        return ""

    cleaned_text = "".join(filter(str.isalnum, text)).upper()

    # 8 lub więcej znaków, zakładamy, że to błąd odczytu ramki i usuwamy pierwszy znak.
    if len(cleaned_text) >= 8 and (cleaned_text[0].isdigit() or cleaned_text[0] == "I"):
        cleaned_text = cleaned_text[1:]

    if len(cleaned_text) == 7:
        correction_map = {"0": "O", "1": "I", "5": "S", "6": "G", "8": "B"}
        final_text = ""
        for i, char in enumerate(cleaned_text):
            is_letter_position = i < 2
            if is_letter_position and char in correction_map:
                final_text += correction_map[char]
            else:
                final_text += char
        return final_text

    return cleaned_text


def get_ground_truth(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    box_element = root.find("box")
    true_box = [
        int(float(box_element.get("xtl"))),
        int(float(box_element.get("ytl"))),
        int(float(box_element.get("xbr"))),
        int(float(box_element.get("ybr"))),
    ]
    true_plate_number = root.find('.//attribute[@name="plate number"]').text
    return true_plate_number, true_box


def process_plate_with_tesseract(image_path):
    results = model(image_path, verbose=False)
    if not results[0].boxes:
        return None, None, None

    predicted_box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    image = cv2.imread(image_path)
    plate_crop = image[
        predicted_box[1] : predicted_box[3], predicted_box[0] : predicted_box[2]
    ]

    # Przycinanie krawędzi
    height, width, _ = plate_crop.shape
    TOP_CROP_PERCENT, BOTTOM_CROP_PERCENT = 0.01, 0.005
    LEFT_CROP_PERCENT, RIGHT_CROP_PERCENT = 0.09, 0.025
    start_y, end_y = int(height * TOP_CROP_PERCENT), int(
        height * (1 - BOTTOM_CROP_PERCENT)
    )
    start_x, end_x = int(width * LEFT_CROP_PERCENT), int(
        width * (1 - RIGHT_CROP_PERCENT)
    )
    trimmed_plate_crop = plate_crop[start_y:end_y, start_x:end_x]

    # Konwersja na skalę szarości i wstępne wygładzanie
    gray_plate = cv2.cvtColor(trimmed_plate_crop, cv2.COLOR_BGR2GRAY)

    # Redukowanie szumu, zachowując ostre krawędzie znaków
    smoothed_plate = cv2.bilateralFilter(gray_plate, 9, 75, 75)

    # Poprawa kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_plate = clahe.apply(smoothed_plate)

    # Binaryzacja metodą Otsu (białe znaki na czarnym tle)
    _, binary_plate_inv = cv2.threshold(
        contrast_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Czyszczenie i wypełnianie znaków
    kernel = np.ones((3, 3), np.uint8)
    cleaned_plate = cv2.morphologyEx(binary_plate_inv, cv2.MORPH_CLOSE, kernel)

    # Odwracamy kolory do odczytu
    final_plate_for_ocr = cv2.bitwise_not(cleaned_plate)

    h, w = final_plate_for_ocr.shape
    scale_factor = 4
    upscaled_plate = cv2.resize(
        final_plate_for_ocr,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    # Konfiguracja i odczyt Tesseracta
    config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    raw_text = pytesseract.image_to_string(upscaled_plate, lang="pol", config=config)
    final_text = post_process_text(raw_text)

    return final_text, predicted_box, upscaled_plate


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:

    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalize accuracy: 60% → 0.0, 100% → 1.0
    accuracy_norm = (accuracy_percent - 60) / 40
    # Normalize time: 60s → 0.0, 10s → 1.0
    time_norm = (60 - processing_time_sec) / 50
    # Compute weighted score
    score = 0.7 * accuracy_norm + 0.3 * time_norm

    grade = 2.0 + 3.0 * score
    # Round to the nearest 0.5
    return round(grade * 2) / 2


# --- Główny blok ewaluacji ---
if __name__ == "__main__":
    test_images_dir = "dataset/images/test"
    test_annotations_dir = "dataset/annotations/test"

    os.makedirs(os.path.dirname(DIAGNOSTIC_CSV_PATH), exist_ok=True)
    os.makedirs(CROPPED_PLATES_DIR, exist_ok=True)

    test_image_files = [
        f for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))
    ]

    if not test_image_files:
        print("BŁĄD: Folder ze zdjęciami testowymi jest pusty!")
        exit()

    correct_ocr_predictions = 0
    total_iou = 0
    detected_count = 0
    processing_times = []

    with open(DIAGNOSTIC_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Nazwa Pliku", "Prawidłowy Tekst", "Wykryty Tekst", "Czy Poprawny?", "IoU"]
        )

        print(f"Rozpoczynam ewaluację na {len(test_image_files)} obrazach testowych...")
        for image_filename in tqdm(test_image_files, desc="Ewaluacja"):
            image_path = os.path.join(test_images_dir, image_filename)
            xml_path = os.path.join(
                test_annotations_dir, os.path.splitext(image_filename)[0] + ".xml"
            )

            ground_truth_text, ground_truth_box = get_ground_truth(xml_path)

            start_time = time.time()
            predicted_text, predicted_box, final_processed_img = (
                process_plate_with_tesseract(image_path)
            )
            processing_times.append(time.time() - start_time)

            iou = 0.0
            is_correct_str = "NIE"

            if predicted_box is not None:
                detected_count += 1
                iou = calculate_iou(ground_truth_box, predicted_box)
                total_iou += iou

                if final_processed_img is not None:
                    output_filename = (
                        f"{os.path.splitext(image_filename)[0]}_processed_crop.jpg"
                    )
                    cv2.imwrite(
                        os.path.join(CROPPED_PLATES_DIR, output_filename),
                        final_processed_img,
                    )

                if predicted_text == ground_truth_text:
                    correct_ocr_predictions += 1
                    is_correct_str = "TAK"

            display_predicted_text = (
                predicted_text if predicted_text is not None else "BRAK_ODCZYTU"
            )
            csv_writer.writerow(
                [
                    image_filename,
                    ground_truth_text,
                    display_predicted_text,
                    is_correct_str,
                    f"{iou:.4f}",
                ]
            )

    # Obliczenia końcowe
    accuracy = (correct_ocr_predictions / len(test_image_files)) * 100
    average_iou = total_iou / detected_count if detected_count > 0 else 0
    total_processing_time = sum(processing_times)
    average_time_per_image = total_processing_time / len(test_image_files)
    time_for_100_images = average_time_per_image * 100
    final_grade = calculate_final_grade(accuracy, time_for_100_images)

    print("")
    print("RAPORT")
    print("--------------------------------------")
    print(f"Liczba obrazów testowych:     {len(test_image_files)}")
    print(f"Liczba poprawnych odczytów:   {correct_ocr_predictions}")
    print(f"Dokładność (Accuracy):        {accuracy:.2f}%")
    print(f"Liczba wykrytych tablic:      {detected_count}")
    print(f"Średnie IoU:                  {average_iou:.4f}")
    print(f"Średni czas na 1 zdjęcie:     {average_time_per_image:.3f} s")
    print(f"Szacowany czas dla 100 zdjęć: {time_for_100_images:.2f} s")
    print("--------------------------------------")
    print(f"Twoja ocena końcowa:          {final_grade:.1f}")
