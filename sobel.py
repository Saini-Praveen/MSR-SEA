import os
import cv2
import numpy as np
import pandas as pd


RAW_FOLDER = "./AUIED3K/test/Raw"
REFERENCE_FOLDER = "./AUIED3K/test/Reference"

METHOD_A_FOLDER = "./CFC_DTRF_AUIED3K/results/CFC-DTRF/test"
METHOD_B_FOLDER = "./UWCNN-AUIED3K/results"

MSRSEA_FOLDER = "./MSR-SEA-AUIED3K/results"

OUTPUT_FOLDER = "Sobel_AUIED3K/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_FOLDER, "sobel_AUIED3K_dataset.csv")
TXT_PATH = os.path.join(OUTPUT_FOLDER, "sobel_AUIED3K_final_results.txt")

IMAGE_SIZE = (256, 256)
EDGE_THRESHOLD = 50
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def calculate_sobel(image):
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return magnitude


def create_edge_map(magnitude, threshold=EDGE_THRESHOLD):
    return (magnitude >= threshold).astype(np.uint8)


def calculate_gradient_mae(test_gradient, reference_gradient):
    return float(np.mean(np.abs(test_gradient - reference_gradient)))


def edge_metrics(test_edges, reference_edges):
    test_edges = test_edges.astype(bool)
    reference_edges = reference_edges.astype(bool)

    tp = np.logical_and(test_edges, reference_edges).sum()
    fp = np.logical_and(test_edges, ~reference_edges).sum()
    fn = np.logical_and(~test_edges, reference_edges).sum()
    union = np.logical_or(test_edges, reference_edges).sum()

    iou = 1.0 if union == 0 else tp / union
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)

    return float(iou), float(precision), float(recall)


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def get_image_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Input folder does not exist:\n{folder}"
        )
    return {
        filename: os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    }


folders = {
    "Raw": RAW_FOLDER,
    "Reference": REFERENCE_FOLDER,
    "Method A": METHOD_A_FOLDER,
    "Method B": METHOD_B_FOLDER,
    "MSR-SEA": MSRSEA_FOLDER,
}

image_dicts = {
    name: get_image_files(folder)
    for name, folder in folders.items()
}

for name, image_dict in image_dicts.items():
    print(f"{name:12s}: {len(image_dict)} images")

common_files = sorted(
    set(image_dicts["Raw"])
    & set(image_dicts["Reference"])
    & set(image_dicts["Method A"])
    & set(image_dicts["Method B"])
    & set(image_dicts["MSR-SEA"])
)

print("\n" + "=" * 80)
print("SOBEL DATASET-LEVEL STRUCTURAL ANALYSIS")
print("=" * 80)
print(f"Complete image pairs : {len(common_files)}")
print(f"Image size           : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
print(f"Edge threshold       : {EDGE_THRESHOLD}")
print("=" * 80)

if not common_files:
    raise RuntimeError(
        "No common image filenames were found across all five folders."
    )


methods = ("Raw", "Method A", "Method B", "MSR-SEA")
results = []
failed_images = []

for index, filename in enumerate(common_files, start=1):
    try:
        reference_image = load_image(
            image_dicts["Reference"][filename]
        )
        reference_gradient = calculate_sobel(reference_image)
        reference_edges = create_edge_map(reference_gradient)

        row = {"Image": filename}

        for method in methods:
            image = load_image(image_dicts[method][filename])
            gradient = calculate_sobel(image)
            edges = create_edge_map(gradient)

            gradient_mae = calculate_gradient_mae(
                gradient, reference_gradient
            )
            edge_iou, edge_precision, edge_recall = edge_metrics(
                edges, reference_edges
            )
            edge_f1 = calculate_f1(edge_precision, edge_recall)

            prefix = method.replace("-", "").replace(" ", "")
            row.update({
                f"{prefix}_Gradient_MAE": gradient_mae,
                f"{prefix}_Edge_IoU": edge_iou,
                f"{prefix}_Edge_Precision": edge_precision,
                f"{prefix}_Edge_Recall": edge_recall,
                f"{prefix}_Edge_F1": edge_f1,
            })

        results.append(row)

        if index % 100 == 0 or index == len(common_files):
            print(f"Processed {index}/{len(common_files)} images")

    except Exception as error:
        failed_images.append((filename, str(error)))
        print(f"[WARNING] Failed: {filename} -> {error}")


df = pd.DataFrame(results)
if df.empty:
    raise RuntimeError("No valid image pairs were processed.")


dataset_metrics = {}

for method in methods:
    prefix = method.replace("-", "").replace(" ", "")

    gradient_mae_avg = float(df[f"{prefix}_Gradient_MAE"].mean())
    edge_iou_avg = float(df[f"{prefix}_Edge_IoU"].mean())
    precision_avg = float(df[f"{prefix}_Edge_Precision"].mean())
    recall_avg = float(df[f"{prefix}_Edge_Recall"].mean())
    edge_f1 = calculate_f1(precision_avg, recall_avg)

    dataset_metrics[method] = {
        "gradient_mae": gradient_mae_avg,
        "edge_iou": edge_iou_avg,
        "edge_precision": precision_avg,
        "edge_recall": recall_avg,
        "edge_f1": edge_f1,
    }


df.to_csv(CSV_PATH, index=False)


with open(TXT_PATH, "w", encoding="utf-8") as file:
    file.write("=" * 90 + "\n")
    file.write("SOBEL DATASET-LEVEL STRUCTURAL ANALYSIS\n")
    file.write("=" * 90 + "\n\n")

    for name in ("Raw", "Reference", "Method A", "Method B", "MSR-SEA"):
        file.write(
            f"Number of {name} images found : "
            f"{len(image_dicts[name])}\n"
        )

    file.write(f"Number of complete image pairs : {len(common_files)}\n")
    file.write(f"Number of valid image pairs    : {len(df)}\n")
    file.write(f"Number of failed images         : {len(failed_images)}\n\n")
    file.write(f"Image size                      : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n")
    file.write(f"Edge threshold                  : {EDGE_THRESHOLD}\n\n")

    file.write("=" * 90 + "\n")
    file.write("DATASET-LEVEL AVERAGE RESULTS\n")
    file.write("=" * 90 + "\n\n")
    file.write(
        "Gradient MAE = mean absolute difference between test and "
        "reference Sobel gradient magnitudes.\n"
    )
    file.write(
        "Edge IoU = intersection-over-union between test and "
        "reference binary edge maps.\n"
    )
    file.write(
        "Edge F1-score = F1 calculated from dataset-level average "
        "Edge Precision and Edge Recall.\n\n"
    )

    file.write(
        f"{'Metric':<22}"
        f"{'Raw':>12}"
        f"{'Method A':>12}"
        f"{'Method B':>12}"
        f"{'MSR-SEA':>12}\n"
    )
    file.write("-" * 70 + "\n")

    table_rows = (
        ("Gradient MAE", "gradient_mae"),
        ("Edge IoU", "edge_iou"),
        ("Edge F1-score", "edge_f1"),
    )

    for label, metric_key in table_rows:
        values = [
            dataset_metrics[method][metric_key]
            for method in methods
        ]
        file.write(
            f"{label:<22}"
            f"{values[0]:>12.3f}"
            f"{values[1]:>12.3f}"
            f"{values[2]:>12.3f}"
            f"{values[3]:>12.3f}\n"
        )

    file.write("\n" + "=" * 90 + "\n")
    file.write("DETAILED DATASET-LEVEL METRICS\n")
    file.write("=" * 90 + "\n\n")

    for method in methods:
        metrics = dataset_metrics[method]
        file.write(f"{method}\n")
        file.write("-" * 50 + "\n")
        file.write(
            f"Gradient MAE   : {metrics['gradient_mae']:.6f}\n"
        )
        file.write(
            f"Edge IoU       : {metrics['edge_iou']:.6f}\n"
        )
        file.write(
            f"Edge Precision : {metrics['edge_precision']:.6f}\n"
        )
        file.write(
            f"Edge Recall    : {metrics['edge_recall']:.6f}\n"
        )
        file.write(
            f"Edge F1-score  : {metrics['edge_f1']:.6f}\n\n"
        )

    if failed_images:
        file.write("=" * 90 + "\n")
        file.write("FAILED IMAGES\n")
        file.write("=" * 90 + "\n\n")
        for filename, error in failed_images:
            file.write(f"{filename} : {error}\n")


print("\n" + "=" * 90)
print("SOBEL DATASET ANALYSIS COMPLETED")
print("=" * 90)
print(f"Valid image pairs processed : {len(df)}")
print(f"Failed images               : {len(failed_images)}")
print(f"CSV saved to                : {CSV_PATH}")
print(f"TXT saved to                : {TXT_PATH}")

print("\nDataset-level averages:")
print("-" * 90)
print(
    f"{'Metric':<22}"
    f"{'Raw':>12}"
    f"{'Method A':>12}"
    f"{'Method B':>12}"
    f"{'MSR-SEA':>12}"
)
for label, metric_key in table_rows:
    values = [
        dataset_metrics[method][metric_key]
        for method in methods
    ]
    print(
        f"{label:<22}"
        f"{values[0]:>12.3f}"
        f"{values[1]:>12.3f}"
        f"{values[2]:>12.3f}"
        f"{values[3]:>12.3f}"
    )
print("=" * 90)
