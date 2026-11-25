import os
import glob
import cv2
import csv
import numpy as np


RAW_FOLDER = './AUIED3K/test/Raw'
REFERENCE_FOLDER = './AUIED3K/test/Reference'

METHOD_A_FOLDER = './CFC_DTRF_AUIED3K/results/CFC-DTRF/test'  
METHOD_B_FOLDER = './UWCNN-AUIED3K/results'  

MSRSEA_FOLDER = './MSR-SEA-AUIED3K/results'

OUTPUT_FOLDER = 'SIFT_AUIED3K/results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_FOLDER, 'sift_feature_correspondence.csv')
TXT_PATH = os.path.join(OUTPUT_FOLDER, 'final_sift_metrics.txt')

IMAGE_SIZE = (256, 256)
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 5.0

sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_image_files(folder):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files.sort()
    return image_files


def extract_sift(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return (None, [], None)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if keypoints is None:
        keypoints = []
    return (image, keypoints, descriptors)


def calculate_correspondence(test_descriptors, test_keypoints,
                             reference_descriptors, reference_keypoints):
    if test_keypoints is None:
        test_keypoints = []
    if reference_keypoints is None:
        reference_keypoints = []

    num_test_keypoints = len(test_keypoints)
    num_reference_keypoints = len(reference_keypoints)

    result = {
        'test_keypoints': num_test_keypoints,
        'reference_keypoints': num_reference_keypoints,
        'good_matches': 0,
        'mean_distance': np.nan,
        'homography_inliers': 0,
        'inlier_ratio': 0.0,
        'matching_status': 'OK'
    }

    if test_descriptors is None:
        result['matching_status'] = 'NO_TEST_DESCRIPTORS'
        return result
    if reference_descriptors is None:
        result['matching_status'] = 'NO_REFERENCE_DESCRIPTORS'
        return result
    if len(test_descriptors) < 2:
        result['matching_status'] = f'INSUFFICIENT_TEST_DESCRIPTORS_{len(test_descriptors)}'
        return result
    if len(reference_descriptors) < 2:
        result['matching_status'] = f'INSUFFICIENT_REFERENCE_DESCRIPTORS_{len(reference_descriptors)}'
        return result

    try:
        matches = flann.knnMatch(test_descriptors, reference_descriptors, k=2)
    except cv2.error as e:
        print('FLANN matching failed.')
        print(f'Test descriptors = {len(test_descriptors)}')
        print(f'Reference descriptors = {len(reference_descriptors)}')
        print(f'OpenCV error: {e}')
        result['matching_status'] = 'FLANN_ERROR'
        return result

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    num_good_matches = len(good_matches)
    result['good_matches'] = num_good_matches

    if num_good_matches > 0:
        result['mean_distance'] = float(np.mean([match.distance for match in good_matches]))

    if num_good_matches < 4:
        if num_good_matches > 0:
            result['matching_status'] = 'INSUFFICIENT_MATCHES_FOR_HOMOGRAPHY'
        else:
            result['matching_status'] = 'NO_GOOD_MATCHES'
        return result

    src_points = np.float32(
        [test_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, RANSAC_THRESHOLD
        )
    except cv2.error as e:
        print('Homography estimation failed.')
        print(f'OpenCV error: {e}')
        result['matching_status'] = 'HOMOGRAPHY_ERROR'
        return result

    if mask is not None:
        mask = mask.ravel()
        num_inliers = int(np.sum(mask))
        result['homography_inliers'] = num_inliers
        result['inlier_ratio'] = num_inliers / num_good_matches * 100.0
        result['matching_status'] = 'OK'
    else:
        result['matching_status'] = 'HOMOGRAPHY_FAILED_NO_MASK'

    return result


folders = {
    'Raw': RAW_FOLDER,
    'Reference': REFERENCE_FOLDER,
    'Method A': METHOD_A_FOLDER,
    'Method B': METHOD_B_FOLDER,
    'MSR-SEA': MSRSEA_FOLDER
}

for name, folder in folders.items():
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f'{name} folder does not exist or is not a directory:\\n{folder}\\n'
            f'Please correct the corresponding input path at the top of sift.py.'
        )

image_lists = {name: get_image_files(folder) for name, folder in folders.items()}

for name, paths in image_lists.items():
    print(f'{name:12s}: {len(paths)} images')
    if len(paths) == 0:
        raise RuntimeError(
            f'No images found in {name} folder:\\n{folders[name]}\\n'
            f'Check the path and make sure images are directly inside this folder.'
        )

image_dicts = {
    name: {os.path.basename(path): path for path in paths}
    for name, paths in image_lists.items()
}

image_names = sorted(image_dicts['Raw'].keys())
print(f'\nTotal raw test images to process: {len(image_names)}')

results = []
model_prefixes = {
    'Raw': 'Raw',
    'Method A': 'MethodA',
    'Method B': 'MethodB',
    'MSR-SEA': 'MSRSEA'
}

counters = {
    'images_seen': 0,
    'images_processed': 0,
    'missing_reference': 0,
    'missing_method_a': 0,
    'missing_method_b': 0,
    'missing_msrsea': 0,
    'read_error_raw': 0,
    'read_error_reference': 0,
    'read_error_method_a': 0,
    'read_error_method_b': 0,
    'read_error_msrsea': 0,
    'reference_insufficient_descriptors': 0
}

for prefix in ('raw', 'method_a', 'method_b', 'msrsea'):
    counters[f'{prefix}_insufficient_descriptors'] = 0
    counters[f'{prefix}_flann_errors'] = 0
    counters[f'{prefix}_homography_errors'] = 0


def update_matching_counters(result, model_prefix):
    status = result['matching_status']
    prefix = {
        'Raw': 'raw',
        'Method A': 'method_a',
        'Method B': 'method_b',
        'MSR-SEA': 'msrsea'
    }[model_prefix]

    if status.startswith('INSUFFICIENT_TEST_DESCRIPTORS'):
        counters[f'{prefix}_insufficient_descriptors'] += 1
    elif status.startswith('INSUFFICIENT_REFERENCE_DESCRIPTORS'):
        counters['reference_insufficient_descriptors'] += 1
    elif status == 'FLANN_ERROR':
        counters[f'{prefix}_flann_errors'] += 1
    elif status == 'HOMOGRAPHY_ERROR':
        counters[f'{prefix}_homography_errors'] += 1


missing_counter = {
    'Reference': 'missing_reference',
    'Method A': 'missing_method_a',
    'Method B': 'missing_method_b',
    'MSR-SEA': 'missing_msrsea'
}
read_error_counter = {
    'Raw': 'read_error_raw',
    'Reference': 'read_error_reference',
    'Method A': 'read_error_method_a',
    'Method B': 'read_error_method_b',
    'MSR-SEA': 'read_error_msrsea'
}

for index, filename in enumerate(image_names, start=1):
    counters['images_seen'] += 1
    print(f'\n[{index}/{len(image_names)}] Processing: {filename}')

    required_names = ('Reference', 'Method A', 'Method B', 'MSR-SEA')
    missing = [name for name in required_names if filename not in image_dicts[name]]
    if missing:
        for name in missing:
            print(f'{name} image missing: {filename}')
            if name in missing_counter:
                counters[missing_counter[name]] += 1
        continue

    paths = {
        'Raw': image_dicts['Raw'][filename],
        'Reference': image_dicts['Reference'][filename],
        'Method A': image_dicts['Method A'][filename],
        'Method B': image_dicts['Method B'][filename],
        'MSR-SEA': image_dicts['MSR-SEA'][filename]
    }

    extracted = {name: extract_sift(path) for name, path in paths.items()}

    read_failed = False
    for name, (image, keypoints, descriptors) in extracted.items():
        if image is None:
            print(f'Unable to read {name} image.')
            counters[read_error_counter[name]] += 1
            read_failed = True
    if read_failed:
        continue

    reference_image, reference_kp, reference_des = extracted['Reference']
    image_results = {}

    for model_name in ('Raw', 'Method A', 'Method B', 'MSR-SEA'):
        _, keypoints, descriptors = extracted[model_name]
        result = calculate_correspondence(
            descriptors, keypoints, reference_des, reference_kp
        )
        image_results[model_name] = result
        update_matching_counters(result, model_name)

    row = {
        'Image': filename,
        'Reference_Keypoints': len(reference_kp)
    }

    for model_name, prefix in model_prefixes.items():
        result = image_results[model_name]
        row.update({
            f'{prefix}_Keypoints': result['test_keypoints'],
            f'{prefix}_Reference_Keypoints': result['reference_keypoints'],
            f'{prefix}_Good_Matches': result['good_matches'],
            f'{prefix}_Mean_Descriptor_Distance': result['mean_distance'],
            f'{prefix}_Homography_Inliers': result['homography_inliers'],
            f'{prefix}_Inlier_Ratio_Percent': result['inlier_ratio'],
            f'{prefix}_Matching_Status': result['matching_status']
        })

    results.append(row)
    counters['images_processed'] += 1

    for model_name in ('Raw', 'Method A', 'Method B', 'MSR-SEA'):
        result = image_results[model_name]
        print(
            f"  {model_name:8s}: KP={result['test_keypoints']}, "
            f"Good={result['good_matches']}, "
            f"Inliers={result['homography_inliers']}, "
            f"IR={result['inlier_ratio']:.2f}%, "
            f"Status={result['matching_status']}"
        )
    print(f'  Reference: KP={len(reference_kp)}')


if len(results) == 0:
    print('\nNo valid image pairs were processed. Check all input paths and filenames.')
    raise SystemExit


csv_columns = ['Image', 'Reference_Keypoints']
for prefix in model_prefixes.values():
    csv_columns.extend([
        f'{prefix}_Keypoints',
        f'{prefix}_Reference_Keypoints',
        f'{prefix}_Good_Matches',
        f'{prefix}_Mean_Descriptor_Distance',
        f'{prefix}_Homography_Inliers',
        f'{prefix}_Inlier_Ratio_Percent',
        f'{prefix}_Matching_Status'
    ])

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(results)


def get_values(key):
    values = []
    for result in results:
        value = result[key]
        if value is None:
            continue
        try:
            value = float(value)
            if np.isfinite(value):
                values.append(value)
        except (TypeError, ValueError):
            pass
    return np.array(values, dtype=np.float64)


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


models = {
    'Raw': {
        'keypoints': 'Raw_Keypoints',
        'reference_keypoints': 'Raw_Reference_Keypoints',
        'good_matches': 'Raw_Good_Matches',
        'homography_inliers': 'Raw_Homography_Inliers'
    },
    'Method A': {
        'keypoints': 'MethodA_Keypoints',
        'reference_keypoints': 'MethodA_Reference_Keypoints',
        'good_matches': 'MethodA_Good_Matches',
        'homography_inliers': 'MethodA_Homography_Inliers'
    },
    'Method B': {
        'keypoints': 'MethodB_Keypoints',
        'reference_keypoints': 'MethodB_Reference_Keypoints',
        'good_matches': 'MethodB_Good_Matches',
        'homography_inliers': 'MethodB_Homography_Inliers'
    },
    'MSR-SEA': {
        'keypoints': 'MSRSEA_Keypoints',
        'reference_keypoints': 'MSRSEA_Reference_Keypoints',
        'good_matches': 'MSRSEA_Good_Matches',
        'homography_inliers': 'MSRSEA_Homography_Inliers'
    }
}

dataset_metrics = {}
for model_name, keys in models.items():
    avg_keypoints = safe_mean(get_values(keys['keypoints']))
    avg_reference_keypoints = safe_mean(get_values(keys['reference_keypoints']))
    avg_good_matches = safe_mean(get_values(keys['good_matches']))
    avg_homography_inliers = safe_mean(get_values(keys['homography_inliers']))

    dataset_metrics[model_name] = {
        'keypoints': int(np.rint(avg_keypoints)),
        'reference_keypoints': int(np.rint(avg_reference_keypoints)),
        'good_matches': int(np.rint(avg_good_matches)),
        'homography_inliers': int(np.rint(avg_homography_inliers))
    }

for model_name in models:
    metrics = dataset_metrics[model_name]
    avg_good_matches = metrics['good_matches']
    avg_inliers = metrics['homography_inliers']
    avg_reference_keypoints = metrics['reference_keypoints']

    if avg_good_matches > 0:
        inlier_precision = avg_inliers / avg_good_matches
    else:
        inlier_precision = 0.0

    if avg_reference_keypoints > 0:
        inlier_recall = avg_inliers / avg_reference_keypoints
    else:
        inlier_recall = 0.0

    if inlier_precision + inlier_recall > 0:
        inlier_f1 = (
            2.0 * inlier_precision * inlier_recall
            / (inlier_precision + inlier_recall)
        )
    else:
        inlier_f1 = 0.0

    metrics['inlier_precision'] = inlier_precision
    metrics['inlier_recall'] = inlier_recall
    metrics['inlier_f1'] = inlier_f1


with open(TXT_PATH, 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('SIFT FEATURE CORRESPONDENCE ANALYSIS\n')
    f.write('=' * 80 + '\n\n')

    for name in ('Raw', 'Reference', 'Method A', 'Method B', 'MSR-SEA'):
        f.write(f'Number of {name} images found: {len(image_lists[name])}\n')
    f.write(f'Number of valid image pairs processed: {len(results)}\n\n')

    f.write(f'Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n')
    f.write(f'Lowe ratio threshold: {RATIO_THRESHOLD}\n')
    f.write(f'RANSAC threshold: {RANSAC_THRESHOLD}\n\n')

    f.write('=' * 80 + '\n')
    f.write('PROCESSING / ROBUSTNESS SUMMARY\n')
    f.write('=' * 80 + '\n\n')
    f.write(f"Images seen                       : {counters['images_seen']}\n")
    f.write(f"Images successfully processed     : {counters['images_processed']}\n")
    f.write(f"Missing reference images           : {counters['missing_reference']}\n")
    f.write(f"Missing Method A images            : {counters['missing_method_a']}\n")
    f.write(f"Missing Method B images            : {counters['missing_method_b']}\n")
    f.write(f"Missing MSR-SEA images             : {counters['missing_msrsea']}\n")
    f.write(f"Raw read errors                    : {counters['read_error_raw']}\n")
    f.write(f"Reference read errors              : {counters['read_error_reference']}\n")
    f.write(f"Method A read errors               : {counters['read_error_method_a']}\n")
    f.write(f"Method B read errors               : {counters['read_error_method_b']}\n")
    f.write(f"MSR-SEA read errors                : {counters['read_error_msrsea']}\n")
    for prefix, label in (
        ('raw', 'Raw'), ('method_a', 'Method A'),
        ('method_b', 'Method B'), ('msrsea', 'MSR-SEA')
    ):
        f.write(
            f'{label} insufficient descriptors'.ljust(36)
            + f": {counters[f'{prefix}_insufficient_descriptors']}\n"
        )
    f.write(
        f"Reference insufficient descriptors: "
        f"{counters['reference_insufficient_descriptors']}\n"
    )
    for prefix, label in (
        ('raw', 'Raw'), ('method_a', 'Method A'),
        ('method_b', 'Method B'), ('msrsea', 'MSR-SEA')
    ):
        f.write(
            f'{label} FLANN errors'.ljust(36)
            + f": {counters[f'{prefix}_flann_errors']}\n"
        )
    for prefix, label in (
        ('raw', 'Raw'), ('method_a', 'Method A'),
        ('method_b', 'Method B'), ('msrsea', 'MSR-SEA')
    ):
        f.write(
            f'{label} homography errors'.ljust(36)
            + f": {counters[f'{prefix}_homography_errors']}\n"
        )

    f.write('\n' + '=' * 80 + '\n')
    f.write('DATASET-LEVEL FEATURE CORRESPONDENCE RESULTS\n')
    f.write('=' * 80 + '\n\n')

    for model_name in models:
        metrics = dataset_metrics[model_name]
        f.write('-' * 80 + '\n')
        f.write(f'{model_name.upper()}\n')
        f.write('-' * 80 + '\n')
        f.write(f"Average SIFT Keypoints              : {metrics['keypoints']}\n")
        f.write(f"Average Reference Keypoints         : {metrics['reference_keypoints']}\n")
        f.write(f"Average Good Matches                : {metrics['good_matches']}\n")
        f.write(f"Average Homography Inliers          : {metrics['homography_inliers']}\n\n")

    f.write('=' * 80 + '\n')
    f.write('INLIER-BASED CORRESPONDENCE METRICS\n')
    f.write('=' * 80 + '\n\n')
    f.write('Inlier Precision = Average Homography Inliers / Average Good Matches\n\n')
    f.write('Inlier Recall = Average Homography Inliers / Average Reference Keypoints\n\n')
    f.write(
        'Inlier F1-score (IF1) = 2 * Inlier Precision * Inlier Recall '
        '/ (Inlier Precision + Inlier Recall)\n\n'
    )
    f.write(
        'The final Inlier Precision, Inlier Recall and Inlier F1-score are '
        'computed using the whole-number dataset average counts reported above.\n\n'
    )

    for model_name in models:
        metrics = dataset_metrics[model_name]
        f.write('-' * 80 + '\n')
        f.write(f'{model_name.upper()}\n')
        f.write('-' * 80 + '\n')
        f.write(f"Inlier Precision                   : {metrics['inlier_precision']:.6f}\n")
        f.write(f"Inlier Recall                      : {metrics['inlier_recall']:.6f}\n")
        f.write(f"Inlier F1-score (IF1)               : {metrics['inlier_f1']:.6f}\n\n")

    f.write('=' * 80 + '\n')
    f.write('SUMMARY TABLE\n')
    f.write('=' * 80 + '\n\n')
    f.write(
        f"{'Metric':<25}"
        f"{'Raw':>10}"
        f"{'Method A':>12}"
        f"{'Method B':>12}"
        f"{'MSR-SEA':>12}\n"
    )
    table_rows = (
        ('SIFT Keypoints', 'keypoints', 'integer'),
        ('Good Matches', 'good_matches', 'integer'),
        ('Homography Inliers', 'homography_inliers', 'integer'),
        ('Inlier F1-score (IF1)', 'inlier_f1', 'float')
    )
    for label, key, value_type in table_rows:
        values = [dataset_metrics[name][key] for name in models]
        if value_type == 'integer':
            formatted = [str(value) for value in values]
        else:
            formatted = [f'{value:.3f}' for value in values]
        f.write(
            f'{label:<25}'
            f'{formatted[0]:>10}'
            f'{formatted[1]:>12}'
            f'{formatted[2]:>12}'
            f'{formatted[3]:>12}\n'
        )

    reference_average = dataset_metrics['Raw']['reference_keypoints']
    f.write('\n' + f'Reference Keypoints: {reference_average}\n')


print('\n' + '=' * 80)
print('SIFT FEATURE ANALYSIS COMPLETE')
print('=' * 80)
print(f"\nImages seen: {counters['images_seen']}")
print(f"Images successfully processed: {counters['images_processed']}")
print(f'\nPer-image CSV: {CSV_PATH}')
print(f'Final metrics TXT: {TXT_PATH}')

print('\n' + '=' * 80)
print('FINAL DATASET RESULTS')
print('=' * 80)

for model_name in models:
    metrics = dataset_metrics[model_name]
    print(f'\n{model_name}')
    print(f"  Average Keypoints: {metrics['keypoints']}")
    print(f"  Average Reference Keypoints: {metrics['reference_keypoints']}")
    print(f"  Average Good Matches: {metrics['good_matches']}")
    print(f"  Average Homography Inliers: {metrics['homography_inliers']}")
    print(f"  Inlier Precision: {metrics['inlier_precision']:.6f}")
    print(f"  Inlier Recall: {metrics['inlier_recall']:.6f}")
    print(f"  Inlier F1-score (IF1): {metrics['inlier_f1']:.6f}")

print('\n' + '=' * 80)
