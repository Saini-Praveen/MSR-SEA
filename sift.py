import os
import glob
import cv2
import csv
import numpy as np

RAW_FOLDER = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Raw'
REFERENCE_FOLDER = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Reference'
UWCNN_FOLDER = '/media/cvblns/NS/PraveenUW/UWCNN-AUIED3K/results'
MSRSEA_FOLDER = '/media/cvblns/NS/PraveenUW/MSR-SEA/MSR-SEA-AUIED3K/results'
OUTPUT_FOLDER = '/media/cvblns/NS/PraveenUW/MSR-SEA/Temp1/SIFT_AUIED3K/output_images'

RAW_OUTPUT = os.path.join(OUTPUT_FOLDER,'raw')
REFERENCE_OUTPUT = os.path.join(OUTPUT_FOLDER,'reference')
UWCNN_OUTPUT = os.path.join(OUTPUT_FOLDER,'uwcnn-enhanced')
MSRSEA_OUTPUT = os.path.join(OUTPUT_FOLDER,'msr-sea-enhanced')

os.makedirs(RAW_OUTPUT,exist_ok=True)
os.makedirs(REFERENCE_OUTPUT,exist_ok=True)
os.makedirs(UWCNN_OUTPUT,exist_ok=True)
os.makedirs(MSRSEA_OUTPUT,exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_FOLDER,'sift_feature_correspondence.csv')
TXT_PATH = os.path.join(OUTPUT_FOLDER,'final_sift_metrics.txt')

IMAGE_SIZE = (256,256)

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 5.0

sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

def get_image_files(folder):
    extensions = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder,ext)))
    image_files.sort()
    return image_files

def extract_sift(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return (None,[],None)
    image = cv2.resize(image,IMAGE_SIZE,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    keypoints,descriptors = sift.detectAndCompute(gray,None)
    if keypoints is None:
        keypoints = []
    return (image,keypoints,descriptors)

def save_keypoint_image(image,keypoints,output_path):
    if image is None:
        return
    if keypoints is None:
        keypoints = []
    visualization = cv2.drawKeypoints(image,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(output_path,visualization)

def calculate_correspondence(test_descriptors,test_keypoints,reference_descriptors,reference_keypoints):
    if test_keypoints is None:
        test_keypoints = []
    if reference_keypoints is None:
        reference_keypoints = []
    num_test_keypoints = len(test_keypoints)
    num_reference_keypoints = len(reference_keypoints)
    result = {'test_keypoints': num_test_keypoints,'reference_keypoints': num_reference_keypoints,'good_matches': 0,'mean_distance': np.nan,'homography_inliers': 0,'inlier_ratio': 0.0,'matching_status': 'OK'}
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
        matches = flann.knnMatch(test_descriptors,reference_descriptors,k=2)
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
        m,n = pair
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
    src_points = np.float32([test_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1,1,2)
    dst_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1,1,2)
    try:
        H,mask = cv2.findHomography(src_points,dst_points,cv2.RANSAC,RANSAC_THRESHOLD)
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
    
raw_images = get_image_files(RAW_FOLDER)
reference_images = get_image_files(REFERENCE_FOLDER)
uwcnn_images = get_image_files(UWCNN_FOLDER)
msrsea_images = get_image_files(MSRSEA_FOLDER)

print(f'Raw images       : {len(raw_images)}')
print(f'Reference images : {len(reference_images)}')
print(f'UWCNN images     : {len(uwcnn_images)}')
print(f'MSR-SEA images   : {len(msrsea_images)}')

raw_dict = {os.path.basename(path): path for path in raw_images}
reference_dict = {os.path.basename(path): path for path in reference_images}
uwcnn_dict = {os.path.basename(path): path for path in uwcnn_images}
msrsea_dict = {os.path.basename(path): path for path in msrsea_images}

image_names = sorted(raw_dict.keys())
print(f'\nTotal raw images to process: {len(image_names)}')

results = []

counters = {'images_seen': 0,'images_processed': 0,'missing_reference': 0,'missing_uwcnn': 0,'missing_msrsea': 0,'read_error_raw': 0,'read_error_reference': 0,'read_error_uwcnn': 0,'read_error_msrsea': 0,'raw_insufficient_descriptors': 0,'uwcnn_insufficient_descriptors': 0,'msrsea_insufficient_descriptors': 0,'reference_insufficient_descriptors': 0,'raw_flann_errors': 0,'uwcnn_flann_errors': 0,'msrsea_flann_errors': 0,'raw_homography_errors': 0,'uwcnn_homography_errors': 0,'msrsea_homography_errors': 0}

def update_matching_counters(result,model_prefix):
    status = result['matching_status']
    if status.startswith('INSUFFICIENT_TEST_DESCRIPTORS'):
        if model_prefix == 'Raw':
            counters['raw_insufficient_descriptors'] += 1
        elif model_prefix == 'UWCNN':
            counters['uwcnn_insufficient_descriptors'] += 1
        elif model_prefix == 'MSR-SEA':
            counters['msrsea_insufficient_descriptors'] += 1
    elif status.startswith('INSUFFICIENT_REFERENCE_DESCRIPTORS'):
        counters['reference_insufficient_descriptors'] += 1
    elif status == 'FLANN_ERROR':
        if model_prefix == 'Raw':
            counters['raw_flann_errors'] += 1
        elif model_prefix == 'UWCNN':
            counters['uwcnn_flann_errors'] += 1
        elif model_prefix == 'MSR-SEA':
            counters['msrsea_flann_errors'] += 1
    elif status == 'HOMOGRAPHY_ERROR':
        if model_prefix == 'Raw':
            counters['raw_homography_errors'] += 1
        elif model_prefix == 'UWCNN':
            counters['uwcnn_homography_errors'] += 1
        elif model_prefix == 'MSR-SEA':
            counters['msrsea_homography_errors'] += 1
            
for index,filename in enumerate(image_names,start=1):
    counters['images_seen'] += 1
    print(f'\n[{index}/{len(image_names)}] Processing: {filename}')
    if filename not in reference_dict:
        print(f'Reference missing: {filename}')
        counters['missing_reference'] += 1
        continue
    if filename not in uwcnn_dict:
        print(f'UWCNN image missing: {filename}')
        counters['missing_uwcnn'] += 1
        continue
    if filename not in msrsea_dict:
        print(f'MSR-SEA image missing: {filename}')
        counters['missing_msrsea'] += 1
        continue
        
    raw_path = raw_dict[filename]
    reference_path = reference_dict[filename]
    uwcnn_path = uwcnn_dict[filename]
    msrsea_path = msrsea_dict[filename]
    
    raw_image,raw_kp,raw_des = extract_sift(raw_path)
    reference_image,reference_kp,reference_des = extract_sift(reference_path)
    uwcnn_image,uwcnn_kp,uwcnn_des = extract_sift(uwcnn_path)
    msrsea_image,msrsea_kp,msrsea_des = extract_sift(msrsea_path)
    
    if raw_image is None:
        print('Unable to read raw image.')
        counters['read_error_raw'] += 1
        continue
    if reference_image is None:
        print('Unable to read reference image.')
        counters['read_error_reference'] += 1
        continue
    if uwcnn_image is None:
        print('Unable to read UWCNN image.')
        counters['read_error_uwcnn'] += 1
        continue
    if msrsea_image is None:
        print('Unable to read MSR-SEA image.')
        counters['read_error_msrsea'] += 1
        continue
        
    save_keypoint_image(raw_image,raw_kp,os.path.join(RAW_OUTPUT,filename))
    save_keypoint_image(reference_image,reference_kp,os.path.join(REFERENCE_OUTPUT,filename))
    save_keypoint_image(uwcnn_image,uwcnn_kp,os.path.join(UWCNN_OUTPUT,filename))
    save_keypoint_image(msrsea_image,msrsea_kp,os.path.join(MSRSEA_OUTPUT,filename))
    raw_result = calculate_correspondence(raw_des,raw_kp,reference_des,reference_kp)
    update_matching_counters(raw_result,'Raw')
    uwcnn_result = calculate_correspondence(uwcnn_des,uwcnn_kp,reference_des,reference_kp)
    update_matching_counters(uwcnn_result,'UWCNN')
    msrsea_result = calculate_correspondence(msrsea_des,msrsea_kp,reference_des,reference_kp)
    update_matching_counters(msrsea_result,'MSR-SEA')
    results.append({'Image': filename,'Raw_Keypoints': raw_result['test_keypoints'],'Raw_Reference_Keypoints': raw_result['reference_keypoints'],'Raw_Good_Matches': 

raw_result['good_matches'],'Raw_Mean_Descriptor_Distance': raw_result['mean_distance'],'Raw_Homography_Inliers': raw_result['homography_inliers'],'Raw_Inlier_Ratio_Percent': raw_result['inlier_ratio'],'Raw_Matching_Status': raw_result['matching_status'],'UWCNN_Keypoints': uwcnn_result['test_keypoints'],'UWCNN_Reference_Keypoints': uwcnn_result['reference_keypoints'],'UWCNN_Good_Matches': uwcnn_result['good_matches'],'UWCNN_Mean_Descriptor_Distance': uwcnn_result['mean_distance'],'UWCNN_Homography_Inliers': uwcnn_result['homography_inliers'],'UWCNN_Inlier_Ratio_Percent': uwcnn_result['inlier_ratio'],'UWCNN_Matching_Status': uwcnn_result['matching_status'],'MSRSEA_Keypoints': msrsea_result['test_keypoints'],'MSRSEA_Reference_Keypoints': msrsea_result['reference_keypoints'],'MSRSEA_Good_Matches': msrsea_result['good_matches'],'MSRSEA_Mean_Descriptor_Distance': msrsea_result['mean_distance'],'MSRSEA_Homography_Inliers': msrsea_result['homography_inliers'],'MSRSEA_Inlier_Ratio_Percent': msrsea_result['inlier_ratio'],'MSRSEA_Matching_Status': msrsea_result['matching_status'],'Reference_Keypoints': len(reference_kp)})
    counters['images_processed'] += 1
    print(f"  Raw     : KP={raw_result['test_keypoints']}, Good={raw_result['good_matches']}, Inliers={raw_result['homography_inliers']}, IR={raw_result['inlier_ratio']:.2f}%, Status={raw_result['matching_status']}")
    print(f"  UWCNN   : KP={uwcnn_result['test_keypoints']}, Good={uwcnn_result['good_matches']}, Inliers={uwcnn_result['homography_inliers']}, IR={uwcnn_result['inlier_ratio']:.2f}%, Status={uwcnn_result['matching_status']}")
    print(f"  MSR-SEA : KP={msrsea_result['test_keypoints']}, Good={msrsea_result['good_matches']}, Inliers={msrsea_result['homography_inliers']}, IR={msrsea_result['inlier_ratio']:.2f}%, Status={msrsea_result['matching_status']}")
    print(f'  Reference: KP={len(reference_kp)}')
if len(results) == 0:
    print('\nNo valid image pairs were processed.')
    raise SystemExit

csv_columns = ['Image','Raw_Keypoints','Raw_Reference_Keypoints','Raw_Good_Matches','Raw_Mean_Descriptor_Distance','Raw_Homography_Inliers','Raw_Inlier_Ratio_Percent','Raw_Matching_Status','UWCNN_Keypoints','UWCNN_Reference_Keypoints','UWCNN_Good_Matches','UWCNN_Mean_Descriptor_Distance','UWCNN_Homography_Inliers','UWCNN_Inlier_Ratio_Percent','UWCNN_Matching_Status','MSRSEA_Keypoints','MSRSEA_Reference_Keypoints','MSRSEA_Good_Matches','MSRSEA_Mean_Descriptor_Distance','MSRSEA_Homography_Inliers','MSRSEA_Inlier_Ratio_Percent','MSRSEA_Matching_Status','Reference_Keypoints']

with open(CSV_PATH,'w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=csv_columns)
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
        except (TypeError,ValueError):
            pass
    return np.array(values,dtype=np.float64)

def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))

models = {'Raw': {'keypoints': 'Raw_Keypoints','reference_keypoints': 'Raw_Reference_Keypoints','good_matches': 'Raw_Good_Matches','homography_inliers': 'Raw_Homography_Inliers'},'UWCNN': {'keypoints': 'UWCNN_Keypoints','reference_keypoints': 'UWCNN_Reference_Keypoints','good_matches': 'UWCNN_Good_Matches','homography_inliers': 'UWCNN_Homography_Inliers'},'MSR-SEA': {'keypoints': 'MSRSEA_Keypoints','reference_keypoints': 'MSRSEA_Reference_Keypoints','good_matches': 'MSRSEA_Good_Matches','homography_inliers': 'MSRSEA_Homography_Inliers'}}

dataset_metrics = {}
for model_name,keys in models.items():
    avg_keypoints = safe_mean(get_values(keys['keypoints']))
    avg_reference_keypoints = safe_mean(get_values(keys['reference_keypoints']))
    avg_good_matches = safe_mean(get_values(keys['good_matches']))
    avg_homography_inliers = safe_mean(get_values(keys['homography_inliers']))
    rounded_keypoints = int(np.rint(avg_keypoints))
    rounded_reference_keypoints = int(np.rint(avg_reference_keypoints))
    rounded_good_matches = int(np.rint(avg_good_matches))
    rounded_homography_inliers = int(np.rint(avg_homography_inliers))
    dataset_metrics[model_name] = {'keypoints': rounded_keypoints,'reference_keypoints': rounded_reference_keypoints,'good_matches': rounded_good_matches,'homography_inliers': rounded_homography_inliers}

for model_name in models:
    avg_good_matches = dataset_metrics[model_name]['good_matches']
    avg_inliers = dataset_metrics[model_name]['homography_inliers']
    avg_reference_keypoints = dataset_metrics[model_name]['reference_keypoints']
    if avg_good_matches > 0:
        inlier_precision = avg_inliers / avg_good_matches
    else:
        inlier_precision = 0.0
    if avg_reference_keypoints > 0:
        inlier_recall = avg_inliers / avg_reference_keypoints
    else:
        inlier_recall = 0.0
    if inlier_precision + inlier_recall > 0:
        inlier_f1 = 2.0 * inlier_precision * inlier_recall / (inlier_precision + inlier_recall)
    else:
        inlier_f1 = 0.0
    dataset_metrics[model_name]['inlier_precision'] = inlier_precision
    dataset_metrics[model_name]['inlier_recall'] = inlier_recall
    dataset_metrics[model_name]['inlier_f1'] = inlier_f1

with open(TXT_PATH,'w') as f:
    f.write('=' * 80 + '\n')
    f.write('SIFT FEATURE CORRESPONDENCE ANALYSIS\n')
    f.write('=' * 80 + '\n\n')
    
    f.write(f'Number of raw images found: {len(raw_images)}\n')
    f.write(f'Number of reference images found: {len(reference_images)}\n')
    f.write(f'Number of UWCNN images found: {len(uwcnn_images)}\n')
    f.write(f'Number of MSR-SEA images found: {len(msrsea_images)}\n')
    f.write(f'Number of valid image pairs processed: {len(results)}\n\n')
    f.write(f'Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n')
    f.write(f'Lowe ratio threshold: {RATIO_THRESHOLD}\n')
    f.write(f'RANSAC threshold: {RANSAC_THRESHOLD}\n\n')
    
    f.write('=' * 80 + '\n')
    f.write('PROCESSING / ROBUSTNESS SUMMARY\n')
    f.write('=' * 80 + '\n\n')
    f.write(f"Images seen                     : {counters['images_seen']}\n")
    f.write(f"Images successfully processed   : {counters['images_processed']}\n")
    f.write(f"Missing reference images        : {counters['missing_reference']}\n")
    f.write(f"Missing UWCNN images            : {counters['missing_uwcnn']}\n")
    f.write(f"Missing MSR-SEA images          : {counters['missing_msrsea']}\n")
    f.write(f"Raw read errors                 : {counters['read_error_raw']}\n")
    f.write(f"Reference read errors           : {counters['read_error_reference']}\n")
    f.write(f"UWCNN read errors               : {counters['read_error_uwcnn']}\n")
    f.write(f"MSR-SEA read errors             : {counters['read_error_msrsea']}\n")
    f.write(f"Raw insufficient descriptors    : {counters['raw_insufficient_descriptors']}\n")
    f.write(f"UWCNN insufficient descriptors  : {counters['uwcnn_insufficient_descriptors']}\n")
    f.write(f"MSR-SEA insufficient descriptors: {counters['msrsea_insufficient_descriptors']}\n")
    f.write(f"Reference insufficient descriptors: {counters['reference_insufficient_descriptors']}\n")
    f.write(f"Raw FLANN errors                : {counters['raw_flann_errors']}\n")
    f.write(f"UWCNN FLANN errors              : {counters['uwcnn_flann_errors']}\n")
    f.write(f"MSR-SEA FLANN errors            : {counters['msrsea_flann_errors']}\n")
    f.write(f"Raw homography errors           : {counters['raw_homography_errors']}\n")
    f.write(f"UWCNN homography errors         : {counters['uwcnn_homography_errors']}\n")
    f.write(f"MSR-SEA homography errors       : {counters['msrsea_homography_errors']}\n\n")
    
    f.write('=' * 80 + '\n')
    f.write('DATASET-LEVEL FEATURE CORRESPONDENCE RESULTS\n')
    f.write('=' * 80 + '\n\n')
    
    for model_name in models:
        metrics = dataset_metrics[model_name]
        f.write('-' * 80 + '\n')
        f.write(f'{model_name.upper()}\n')
        f.write('-' * 80 + '\n')
        f.write(f"Average SIFT Keypoints              : {metrics['keypoints']}\n")
        f.write(f"Average Reference Keypoints         : {metrics['reference_keypoints']}\n")
        f.write(f"Average Good Matches               : {metrics['good_matches']}\n")
        f.write(f"Average Homography Inliers         : {metrics['homography_inliers']}\n")
        f.write('\n')
    f.write('=' * 80 + '\n')
    f.write('INLIER-BASED CORRESPONDENCE METRICS\n')
    f.write('=' * 80 + '\n\n')
    f.write('Inlier Precision = Average Homography Inliers / Average Good Matches\n\n')
    f.write('Inlier Recall = Average Homography Inliers / Average Reference Keypoints\n\n')
    f.write('Inlier F1-score (IF1) = 2 * Inlier Precision * Inlier Recall / (Inlier Precision + Inlier Recall)\n\n')
    f.write('The final Inlier Precision, Inlier Recall and Inlier F1-score are computed using the whole-number dataset average counts reported above.\n\n')
    
    for model_name in models:
        metrics = dataset_metrics[model_name]
        f.write('-' * 80 + '\n')
        f.write(f'{model_name.upper()}\n')
        f.write('-' * 80 + '\n')
        f.write(f"Inlier Precision                  : {metrics['inlier_precision']:.6f}\n")
        f.write(f"Inlier Recall                     : {metrics['inlier_recall']:.6f}\n")
        f.write(f"Inlier F1-score (IF1)              : {metrics['inlier_f1']:.6f}\n")
        f.write('\n')
print('\n' + '=' * 80)
print('SIFT FEATURE ANALYSIS COMPLETE')
print('=' * 80)
print(f"\nImages seen: {counters['images_seen']}")
print(f"Images successfully processed: {counters['images_processed']}")
print(f'\nPer-image CSV:')
print(CSV_PATH)
print(f'\nFinal metrics TXT:')
print(TXT_PATH)
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
