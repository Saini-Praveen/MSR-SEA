import os
import cv2
import numpy as np
import pandas as pd

RAW_FOLDER = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Raw'
UWCNN_FOLDER = '/media/cvblns/NS/PraveenUW/UWCNN-AUIED3K/results'
MSRSEA_FOLDER = '/media/cvblns/NS/PraveenUW/MSR-SEA/MSR-SEA-AUIED3K/results'
REFERENCE_FOLDER = '/media/cvblns/NS/Praveen/UW_Datasets/AUIED3K/Reference'
OUTPUT_FOLDER = '/media/cvblns/NS/PraveenUW/MSR-SEA/Temp1/Sobel_AUIED3K'

IMAGE_SIZE = (256,256)
EDGE_THRESHOLD = 50

GRADIENT_DIRS = {'raw': os.path.join(OUTPUT_FOLDER,'gradient_maps','raw'),'uwcnn': os.path.join(OUTPUT_FOLDER,'gradient_maps','uwcnn'),'msrsea': os.path.join(OUTPUT_FOLDER,'gradient_maps','msrsea'),'reference': os.path.join(OUTPUT_FOLDER,'gradient_maps','reference')}

EDGE_DIRS = {'raw': os.path.join(OUTPUT_FOLDER,'edge_maps','raw'),'uwcnn': os.path.join(OUTPUT_FOLDER,'edge_maps','uwcnn'),'msrsea': os.path.join(OUTPUT_FOLDER,'edge_maps','msrsea'),'reference': os.path.join(OUTPUT_FOLDER,'edge_maps','reference')}

for directory in list(GRADIENT_DIRS.values()) + list(EDGE_DIRS.values()):
    os.makedirs(directory,exist_ok=True)

def calculate_sobel(image):
    image = cv2.resize(image,IMAGE_SIZE)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    sobel_x = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    sobel_y = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitude = cv2.magnitude(sobel_x,sobel_y)
    return magnitude

def normalize_gradient(magnitude):
    normalized = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def create_edge_map(magnitude,threshold=EDGE_THRESHOLD):
    edge_map = (magnitude >= threshold).astype(np.uint8)
    return edge_map

def calculate_gradient_mae(test_gradient,reference_gradient):
    gradient_mae = np.mean(np.abs(test_gradient - reference_gradient))
    return gradient_mae

def edge_metrics(test_edges,reference_edges):
    test_edges = test_edges.astype(bool)
    reference_edges = reference_edges.astype(bool)
    TP = np.logical_and(test_edges,reference_edges).sum()
    FP = np.logical_and(test_edges,~reference_edges).sum()
    FN = np.logical_and(~test_edges,reference_edges).sum()
    union = np.logical_or(test_edges,reference_edges).sum()
    intersection = TP
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    return (iou,precision,recall)

def calculate_f1(precision,recall):
    if precision + recall == 0:
        return 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f'Unable to read image: {path}')
    return image

IMAGE_EXTENSIONS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

raw_files = sorted([f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)])
uwcnn_files = sorted([f for f in os.listdir(UWCNN_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)])
msrsea_files = sorted([f for f in os.listdir(MSRSEA_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)])
reference_files = sorted([f for f in os.listdir(REFERENCE_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)])

common_files = sorted(set(raw_files) & set(uwcnn_files) & set(msrsea_files) & set(reference_files))
print('\n' + '=' * 80)
print('SOBEL DATASET-LEVEL ANALYSIS')
print('=' * 80)
print(f'Raw images found       : {len(raw_files)}')
print(f'UWCNN images found     : {len(uwcnn_files)}')
print(f'MSR-SEA images found   : {len(msrsea_files)}')
print(f'Reference images found : {len(reference_files)}')
print(f'Complete image pairs   : {len(common_files)}')
print(f'Image size             : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}')
print(f'Edge threshold         : {EDGE_THRESHOLD}')
print('=' * 80)

results = []
failed_images = []
for index,filename in enumerate(common_files,start=1):
    try:
        raw_path = os.path.join(RAW_FOLDER,filename)
        uwcnn_path = os.path.join(UWCNN_FOLDER,filename)
        msrsea_path = os.path.join(MSRSEA_FOLDER,filename)
        reference_path = os.path.join(REFERENCE_FOLDER,filename)
        
        raw_image = load_image(raw_path)
        uwcnn_image = load_image(uwcnn_path)
        msrsea_image = load_image(msrsea_path)
        reference_image = load_image(reference_path)
        
        raw_gradient = calculate_sobel(raw_image)
        uwcnn_gradient = calculate_sobel(uwcnn_image)
        msrsea_gradient = calculate_sobel(msrsea_image)
        reference_gradient = calculate_sobel(reference_image)
        
        cv2.imwrite(os.path.join(GRADIENT_DIRS['raw'],filename),normalize_gradient(raw_gradient))
        cv2.imwrite(os.path.join(GRADIENT_DIRS['uwcnn'],filename),normalize_gradient(uwcnn_gradient))
        cv2.imwrite(os.path.join(GRADIENT_DIRS['msrsea'],filename),normalize_gradient(msrsea_gradient))
        cv2.imwrite(os.path.join(GRADIENT_DIRS['reference'],filename),normalize_gradient(reference_gradient))
        
        raw_gradient_mae = calculate_gradient_mae(raw_gradient,reference_gradient)
        uwcnn_gradient_mae = calculate_gradient_mae(uwcnn_gradient,reference_gradient)
        msrsea_gradient_mae = calculate_gradient_mae(msrsea_gradient,reference_gradient)
        
        raw_edges = create_edge_map(raw_gradient)
        uwcnn_edges = create_edge_map(uwcnn_gradient)
        msrsea_edges = create_edge_map(msrsea_gradient)
        reference_edges = create_edge_map(reference_gradient)
        
        cv2.imwrite(os.path.join(EDGE_DIRS['raw'],filename),raw_edges * 255)
        cv2.imwrite(os.path.join(EDGE_DIRS['uwcnn'],filename),uwcnn_edges * 255)
        cv2.imwrite(os.path.join(EDGE_DIRS['msrsea'],filename),msrsea_edges * 255)
        cv2.imwrite(os.path.join(EDGE_DIRS['reference'],filename),reference_edges * 255)
        
        raw_edge_iou,raw_edge_precision,raw_edge_recall = edge_metrics(raw_edges,reference_edges)
        uwcnn_edge_iou,uwcnn_edge_precision,uwcnn_edge_recall = edge_metrics(uwcnn_edges,reference_edges)
        msrsea_edge_iou,msrsea_edge_precision,msrsea_edge_recall = edge_metrics(msrsea_edges,reference_edges)
        
        results.append({'Image': filename,'Raw_Gradient_MAE': raw_gradient_mae,'Raw_Edge_IoU': raw_edge_iou,'Raw_Edge_Precision': raw_edge_precision,'Raw_Edge_Recall': raw_edge_recall,'UWCNN_Gradient_MAE': uwcnn_gradient_mae,'UWCNN_Edge_IoU': uwcnn_edge_iou,'UWCNN_Edge_Precision': uwcnn_edge_precision,'UWCNN_Edge_Recall': uwcnn_edge_recall,'MSRSEA_Gradient_MAE': msrsea_gradient_mae,'MSRSEA_Edge_IoU': msrsea_edge_iou,'MSRSEA_Edge_Precision': msrsea_edge_precision,'MSRSEA_Edge_Recall': msrsea_edge_recall})
        if index % 100 == 0 or index == len(common_files):
            print(f'Processed {index}/{len(common_files)} images')
    except Exception as e:
        failed_images.append((filename,str(e)))
        print(f'[WARNING] Failed: {filename} -> {e}')
df = pd.DataFrame(results)
if len(df) == 0:
    raise RuntimeError('No valid image pairs were processed.')
    
raw_gradient_mae_avg = df['Raw_Gradient_MAE'].mean()
uwcnn_gradient_mae_avg = df['UWCNN_Gradient_MAE'].mean()
msrsea_gradient_mae_avg = df['MSRSEA_Gradient_MAE'].mean()

raw_edge_iou_avg = df['Raw_Edge_IoU'].mean()
uwcnn_edge_iou_avg = df['UWCNN_Edge_IoU'].mean()
msrsea_edge_iou_avg = df['MSRSEA_Edge_IoU'].mean()

raw_precision_avg = df['Raw_Edge_Precision'].mean()
uwcnn_precision_avg = df['UWCNN_Edge_Precision'].mean()
msrsea_precision_avg = df['MSRSEA_Edge_Precision'].mean()

raw_recall_avg = df['Raw_Edge_Recall'].mean()
uwcnn_recall_avg = df['UWCNN_Edge_Recall'].mean()
msrsea_recall_avg = df['MSRSEA_Edge_Recall'].mean()

raw_edge_f1 = calculate_f1(raw_precision_avg,raw_recall_avg)
uwcnn_edge_f1 = calculate_f1(uwcnn_precision_avg,uwcnn_recall_avg)
msrsea_edge_f1 = calculate_f1(msrsea_precision_avg,msrsea_recall_avg)

csv_path = os.path.join(OUTPUT_FOLDER,'sobel_AUIED3K_dataset.csv')
df.to_csv(csv_path,index=False)
txt_path = os.path.join(OUTPUT_FOLDER,'sobel_AUIED3K_final_results.txt')
with open(txt_path,'w') as f:
    f.write('=' * 80 + '\n')
    f.write('SOBEL DATASET-LEVEL STRUCTURAL ANALYSIS\n')
    f.write('=' * 80 + '\n\n')
    f.write(f'Number of Raw images found       : {len(raw_files)}\n')
    f.write(f'Number of UWCNN images found     : {len(uwcnn_files)}\n')
    f.write(f'Number of MSR-SEA images found   : {len(msrsea_files)}\n')
    f.write(f'Number of Reference images found : {len(reference_files)}\n')
    f.write(f'Number of valid image pairs      : {len(df)}\n')
    f.write(f'Number of failed images          : {len(failed_images)}\n\n')
    f.write(f'Image size                       : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n')
    f.write(f'Edge threshold                   : {EDGE_THRESHOLD}\n\n')
    f.write('=' * 80 + '\n')
    f.write('DATASET-LEVEL AVERAGE RESULTS\n')
    f.write('=' * 80 + '\n\n')
    f.write('Gradient MAE = mean absolute difference between test and reference Sobel gradient magnitudes.\n')
    f.write('Edge IoU = intersection-over-union between test and reference binary edge maps.\n')
    f.write('Edge F1-score = F1 calculated from the dataset-level average Edge Precision and Edge Recall.\n\n')
    f.write(f"{'Metric':<25}{'Raw':>15}{'UWCNN':>15}{'MSR-SEA':>15}\n")
    f.write('-' * 70 + '\n')
    f.write(f"{'Gradient MAE':<25}{raw_gradient_mae_avg:>15.6f}{uwcnn_gradient_mae_avg:>15.6f}{msrsea_gradient_mae_avg:>15.6f}\n")
    f.write(f"{'Edge IoU':<25}{raw_edge_iou_avg:>15.6f}{uwcnn_edge_iou_avg:>15.6f}{msrsea_edge_iou_avg:>15.6f}\n")
    f.write(f"{'Edge Precision':<25}{raw_precision_avg:>15.6f}{uwcnn_precision_avg:>15.6f}{msrsea_precision_avg:>15.6f}\n")
    f.write(f"{'Edge Recall':<25}{raw_recall_avg:>15.6f}{uwcnn_recall_avg:>15.6f}{msrsea_recall_avg:>15.6f}\n")
    f.write(f"{'Edge F1-score':<25}{raw_edge_f1:>15.6f}{uwcnn_edge_f1:>15.6f}{msrsea_edge_f1:>15.6f}\n")
    f.write('\n' + '=' * 80 + '\n')
    f.write('FINAL METRICS\n')
    f.write('=' * 80 + '\n\n')
    f.write(f'Raw     : Gradient MAE = {raw_gradient_mae_avg:.3f}, Edge IoU = {raw_edge_iou_avg:.3f}, Edge F1 = {raw_edge_f1:.3f}\n')
    f.write(f'UWCNN   : Gradient MAE = {uwcnn_gradient_mae_avg:.3f}, Edge IoU = {uwcnn_edge_iou_avg:.3f}, Edge F1 = {uwcnn_edge_f1:.3f}\n')
    f.write(f'MSR-SEA : Gradient MAE = {msrsea_gradient_mae_avg:.3f}, Edge IoU = {msrsea_edge_iou_avg:.3f}, Edge F1 = {msrsea_edge_f1:.3f}\n')
    
    if len(failed_images) > 0:
        f.write('\n' + '=' * 80 + '\n')
        f.write('FAILED IMAGES\n')
        f.write('=' * 80 + '\n\n')
        for filename,error in failed_images:
            f.write(f'{filename} : {error}\n')
    f.write('\n' + '=' * 80 + '\n')
print('\n' + '=' * 80)
print('SOBEL DATASET ANALYSIS COMPLETED')
print('=' * 80)
print(f'Valid image pairs processed : {len(df)}')
print(f'Failed images               : {len(failed_images)}')
print(f'CSV saved to                : {csv_path}')
print(f'TXT saved to                : {txt_path}')
print('\nDataset-level averages:')
print('-' * 80)
print(f'Raw     -> Gradient MAE: {raw_gradient_mae_avg:.6f}, Edge IoU: {raw_edge_iou_avg:.6f}, Edge F1: {raw_edge_f1:.6f}')
print(f'UWCNN   -> Gradient MAE: {uwcnn_gradient_mae_avg:.6f}, Edge IoU: {uwcnn_edge_iou_avg:.6f}, Edge F1: {uwcnn_edge_f1:.6f}')
print(f'MSR-SEA -> Gradient MAE: {msrsea_gradient_mae_avg:.6f}, Edge IoU: {msrsea_edge_iou_avg:.6f}, Edge F1: {msrsea_edge_f1:.6f}')
print('=' * 80)
