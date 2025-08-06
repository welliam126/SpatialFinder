import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, 
                            confusion_matrix, recall_score, precision_score,
                            roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

# Configuration parameters
TEST_DATA_PATH = "path/to/test_data"
WEIGHT_SAVE_PATH = "model_weights/"
OUTPUT_PATH = "path/to/output_results/"
PROBABILITY_THRESHOLD = 0.6 

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

class CoreFeatureProcessor:
    def __init__(self, base_color=(240, 240, 240)):
        self.base_color = base_color
        
    def process_image(self, input_img):
        # Create processing mask
        mask = self._create_processing_mask(input_img)
        processed_img = cv2.bitwise_and(input_img, input_img, mask=mask)
        
        # Transform image for analysis
        transformed = self._transform_image(processed_img)
        
        # Identify primary region
        primary = self._locate_primary_region(transformed)
        if primary is None:
            return self._default_results()
        
        # Identify secondary region
        secondary = self._locate_secondary_region(transformed, primary)
        
        # Calculate region metrics
        metrics = self._compute_region_metrics(transformed, primary, secondary)
        
        # Calculate shape properties
        shape_props = self._compute_shape_properties(primary['contour'])
        
        return {
            'status': 'Success',
            'metric1': metrics['metric1'],
            'metric2': metrics['metric2'],
            'metric3': metrics['metric3'],
            'area': primary['area'],
            'center_x': primary['center'][0],
            'center_y': primary['center'][1],
            'prop1': shape_props[0],
            'prop2': shape_props[1]
        }
    
    def _default_results(self):
        return {
            'status': 'Failed: Primary region not found',
            'metric1': 0,
            'metric2': 0,
            'metric3': 0,
            'area': 0,
            'center_x': 0,
            'center_y': 0,
            'prop1': 0,
            'prop2': 0
        }
    
    def _create_processing_mask(self, img):
        lower = np.array([self.base_color[2] - 5, 
                         self.base_color[1] - 5, 
                         self.base_color[0] - 5])
        upper = np.array([self.base_color[2] + 5, 
                         self.base_color[1] + 5, 
                         self.base_color[0] + 5])
        base_mask = cv2.inRange(img, lower, upper)
        return cv2.bitwise_not(base_mask)
    
    def _transform_image(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value_ch = hsv_img[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(value_ch)
        return cv2.medianBlur(enhanced, 5)
    
    def _locate_primary_region(self, img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(main_contour)
        
        return {'center': (cx, cy), 'area': area, 'contour': main_contour}
    
    def _locate_secondary_region(self, img, primary):
        cx, cy = primary['center']
        radius = np.sqrt(primary['area'] / np.pi)
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius * 1.5), 255, -1)
        cv2.circle(mask, (cx, cy), int(radius * 0.8), 0, -1)
        region = cv2.bitwise_and(img, img, mask=mask)
        _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    
    def _compute_region_metrics(self, img, primary, secondary):
        primary_mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(primary_mask, [primary['contour']], -1, 255, -1)
        
        primary_pixels = img[primary_mask == 255]
        secondary_pixels = img[(secondary > 0) & (primary_mask == 0)]
        
        if len(primary_pixels) == 0 or len(secondary_pixels) == 0:
            return {'metric1': 0, 'metric2': 0, 'metric3': 0}
        
        primary_val = np.percentile(primary_pixels, 85)
        secondary_val = np.percentile(secondary_pixels, 15)
        metric1 = primary_val / (secondary_val + 1e-6)
        
        contours, _ = cv2.findContours(secondary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull)
            metric2 = area / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            metric3 = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        else:
            metric2 = 0
            metric3 = 0
        
        return {
            'metric1': float(metric1),
            'metric2': float(metric2),
            'metric3': float(metric3)
        }
    
    def _compute_shape_properties(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        prop2 = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        if len(contour) < 5:
            return 0.0, prop2
        
        M = cv2.moments(contour)
        if M['mu02'] == 0:
            return 0.0, prop2
            
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        mu20 = M['mu20'] / M['m00']**2
        mu02 = M['mu02'] / M['m00']**2
        mu11 = M['mu11'] / M['m00']**2
        mu30 = M['mu30'] / M['m00']**2.5
        mu03 = M['mu03'] / M['m00']**2.5
        
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        
        mu20_prime = mu20 * math.cos(theta)**2 + mu02 * math.sin(theta)**2 + mu11 * math.sin(2*theta)
        mu02_prime = mu20 * math.sin(theta)**2 + mu02 * math.cos(theta)**2 - mu11 * math.sin(2*theta)
        mu30_prime = mu30 * math.cos(theta)**3 - mu03 * math.sin(theta)**3
        
        prop1 = mu30_prime / (mu20_prime**1.5) if mu20_prime > 0 else 0
        
        return float(prop1), float(prop2)

def extract_image_features(image_path, processor):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image {image_path}")
        return [0] * 6
    
    features = processor.process_image(img)
    
    if features['status'] == 'Success':
        return [
            features['area'],
            features['prop2'],
            features['metric1'],
            features['metric2'],
            features['metric3'],
            features['prop1']
        ]
    else:
        print(f"Warning: Feature extraction failed {image_path}")
        return [0] * 6

def calculate_performance_metrics(y_true, y_pred, y_proba, classes):
    """
    Calculate multi-class performance metrics
    """
    results = {}
    
    # Overall metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    class_metrics = {}
    for i, cls in enumerate(classes):
        # Binary classification metrics
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Calculate metrics
        class_metrics[cls] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'auc': roc_auc_score(y_true_binary, y_proba[:, i]),
            'support': sum(y_true == i)
        }
    
    # Macro-averaged metrics
    results['macro_precision'] = np.mean([m['precision'] for m in class_metrics.values()])
    results['macro_recall'] = np.mean([m['recall'] for m in class_metrics.values()])
    results['macro_f1'] = np.mean([m['f1'] for m in class_metrics.values()])
    results['macro_auc'] = np.mean([m['auc'] for m in class_metrics.values()])
    
    results['class_metrics'] = class_metrics
    
    return results

def evaluate_model():
    # 1. Initialize feature processor
    processor = CoreFeatureProcessor()
    
    # 2. Load model and scaler
    model = XGBClassifier()
    model.load_model(os.path.join(WEIGHT_SAVE_PATH, 'tls_classifier.model'))
    
    scaler_params = np.load(
        os.path.join(WEIGHT_SAVE_PATH, 'scaler_params.npy'), 
        allow_pickle=True
    ).item()
    
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']
    scaler.var_ = scaler_params['scale'] ** 2
    
    # 3. Collect test images and labels
    image_paths = []
    labels = []
    
    # Class mapping
    class_mapping = {'class_a': 0, 'class_b': 1, 'class_c': 2}
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    # Traverse test directories
    for class_name in class_mapping.keys():
        class_dir = os.path.join(TEST_DATA_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found {class_dir}")
            continue
            
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                img_path = os.path.join(class_dir, img_file)
                image_paths.append(img_path)
                labels.append(class_mapping[class_name])
    
    if not image_paths:
        print("Error: No images found in test set")
        return
    
    print(f"Found {len(image_paths)} test samples")
    
    # 4. Extract features
    features = []
    failed_paths = []
    
    print("Extracting features from test set...")
    for img_path in tqdm(image_paths, desc="Processing images"):
        feat = extract_image_features(img_path, processor)
        if sum(feat) == 0:  # Feature extraction failed
            failed_paths.append(img_path)
        features.append(feat)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features, columns=[
        'Area', 'ShapeProp2', 'Metric1', 
        'Metric2', 'Metric3', 'ShapeProp1'
    ])
    
    # 5. Feature engineering (consistent with training)
    feature_df['Log_Area'] = np.log1p(feature_df['Area'])
    feature_df['Metric1_Squared'] = feature_df['Metric1']**2
    
    # Feature weighting
    feature_weights = {
        'Area': 1.5,
        'ShapeProp2': 1.3,
        'Log_Area': 1.2,
        'Metric1': 1.0,
        'Metric2': 0.8,
        'Metric3': 0.8,
        'Metric1_Squared': 0.6,
        'ShapeProp1': 0.6
    }
    
    for feat, weight in feature_weights.items():
        if feat in feature_df.columns:
            feature_df[feat] = feature_df[feat] * weight
    
    # 6. Scale features
    X_test_scaled = scaler.transform(feature_df)
    
    # 7. Predict probabilities
    y_proba = model.predict_proba(X_test_scaled)
    
    # 8. Apply probability threshold
    max_proba = np.max(y_proba, axis=1)
    y_pred_index = np.argmax(y_proba, axis=1)
    
    # Create "uncertain" category (index 3)
    uncertain_mask = max_proba < PROBABILITY_THRESHOLD
    y_pred_index[uncertain_mask] = 3
    
    # 9. Save predictions
    results_df = pd.DataFrame({
        'Image_Path': image_paths,
        'True_Label': [reverse_mapping[label] for label in labels],
        'Predicted_Label': [reverse_mapping.get(idx, 'uncertain') for idx in y_pred_index],
        'Prob_class_a': y_proba[:, 0],
        'Prob_class_b': y_proba[:, 1],
        'Prob_class_c': y_proba[:, 2],
        'Max_Probability': max_proba,
        'Is_Uncertain': uncertain_mask
    })
    
    results_csv_path = os.path.join(OUTPUT_PATH, 'test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions saved to: {results_csv_path}")
    
    # 10. Calculate metrics (excluding uncertain samples)
    valid_mask = ~uncertain_mask
    y_true = np.array(labels)
    
    if np.any(valid_mask):
        # Consider only certain predictions
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_index[valid_mask]
        y_proba_valid = y_proba[valid_mask]
        
        # Calculate metrics
        metrics = calculate_performance_metrics(
            y_true_valid, 
            y_pred_valid, 
            y_proba_valid,
            classes=list(class_mapping.keys())
        )
        
        # Print overall metrics
        print("\nTest Set Evaluation (Certain Samples Only):")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"- Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"- Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"- Macro Precision: {metrics['macro_precision']:.4f}")
        
        # Print per-class metrics
        print("\nPer-Class Metrics:")
        for cls, cls_metrics in metrics['class_metrics'].items():
            print(f"\nClass: {cls}")
            print(f"  - Precision: {cls_metrics['precision']:.4f}")
            print(f"  - Recall: {cls_metrics['recall']:.4f}")
            print(f"  - F1: {cls_metrics['f1']:.4f}")
            print(f"  - Specificity: {cls_metrics['specificity']:.4f}")
            print(f"  - AUC: {cls_metrics['auc']:.4f}")
            print(f"  - Support: {cls_metrics['support']}")
        
        # Save metrics to files
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Weighted F1', 'Macro AUC', 'Macro Recall', 'Macro Precision'],
            'Value': [
                metrics['accuracy'],
                metrics['weighted_f1'],
                metrics['macro_auc'],
                metrics['macro_recall'],
                metrics['macro_precision']
            ]
        })
        metrics_df.to_csv(os.path.join(OUTPUT_PATH, 'test_metrics.csv'), index=False)
        
        # Create class metrics dataframe
        class_metrics_list = []
        for cls, m in metrics['class_metrics'].items():
            row = {
                'Class': cls,
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1': m['f1'],
                'Specificity': m['specificity'],
                'AUC': m['auc'],
                'Support': m['support']
            }
            class_metrics_list.append(row)
        
        class_metrics_df = pd.DataFrame(class_metrics_list)
        class_metrics_df.to_csv(os.path.join(OUTPUT_PATH, 'class_metrics.csv'), index=False)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(class_mapping.keys()), 
                    yticklabels=list(class_mapping.keys()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for i, cls in enumerate(class_mapping.keys()):
            fpr, tpr, _ = roc_curve((y_true_valid == i).astype(int), y_proba_valid[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_PATH, 'roc_curve.png'))
        plt.close()
    else:
        print("Warning: All test samples marked as uncertain")
    
    # Print uncertain sample ratio
    print(f"\nUncertain sample ratio: {np.mean(uncertain_mask):.2%}")

if __name__ == "__main__":
    print("="*50)
    print("Test Set Evaluation")
    print(f"Test data path: {TEST_DATA_PATH}")
    print(f"Model path: {WEIGHT_SAVE_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Probability threshold: {PROBABILITY_THRESHOLD}")
    print("="*50)
    
    evaluate_model()
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)