import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from xgboost import XGBClassifier
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration parameters
WEIGHT_SAVE_PATH = "model_weights/"
PROBABILITY_THRESHOLD = 0.6
RANDOM_STATE = 42
TEST_SIZE = 0.2

class CoreFeatureProcessor:
    def __init__(self, base_color=(240, 240, 240)):
        self.base_color = base_color
        
    def process_image(self, input_img):
        # Create processing mask
        mask = self._generate_mask(input_img)
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
    
    def _generate_mask(self, img):
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

def validate_feature_statistics(feature_df, label_df, output_path):
    results = {}
    for group in ['foll', 'fol', 'agg']:
        group_data = feature_df[label_df == group]
        stats = {
            'Area': group_data['Area'].median(),
            'ShapeProp2': group_data['ShapeProp2'].median(),
            'Metric1': group_data['Metric1'].median(),
            'Metric2': group_data['Metric2'].median(),
            'Metric3': group_data['Metric3'].median(),
            'ShapeProp1': group_data['ShapeProp1'].median()
        }
        results[group] = stats
    
    print("\nFeature Statistics:")
    print(f"{'Feature':<15} {'foll':<12} {'fol':<12} {'agg':<12}")
    
    for feature in ['Area', 'ShapeProp2', 'Metric1', 'Metric2', 'Metric3', 'ShapeProp1']:
        row = f"{feature:<15}"
        for group in ['foll', 'fol', 'agg']:
            row += f"{results[group][feature]:<12.2f}"
        print(row)
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['Area', 'ShapeProp2', 'Metric1', 'Metric2', 'Metric3', 'ShapeProp1']):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=label_df, y=feature_df[feature])
        plt.title(feature)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_distributions.png'))
    plt.close()
    
    return results

def train_tls_classifier(input_path, output_path):
    processor = CoreFeatureProcessor()
    
    image_paths = []
    labels = []
    
    for class_name in ['class_a', class_b', 'class_c']:
        class_dir = os.path.join(input_path, class_name)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_name)
    
    features = []
    failed_paths = []
    
    print("Processing images...")
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        feat = extract_image_features(img_path, processor)
        if sum(feat) == 0:
            failed_paths.append(img_path)
        features.append(feat)
    
    print(f"\nSuccessful analyses: {len(features) - len(failed_paths)}/{len(features)}")
    
    feature_df = pd.DataFrame(features, columns=[
        'Area', 'ShapeProp2', 'Metric1', 
        'Metric2', 'Metric3', 'ShapeProp1'
    ])
    label_df = pd.Series(labels, name='Label')
    
    # Feature engineering
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
    
    validate_feature_statistics(feature_df, label_df, output_path)
    
    label_map = {'foll': 0, 'fol': 1, 'agg': 2}
    y_numeric = label_df.map(label_map)
    
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y_numeric, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_numeric
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    print("\nTraining classification model...")
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    
    y_proba = model.predict_proba(X_test_scaled)
    max_proba = np.max(y_proba, axis=1)
    y_pred_index = np.argmax(y_proba, axis=1)
    
    uncertain_mask = max_proba < PROBABILITY_THRESHOLD
    y_pred_index[uncertain_mask] = 3
    
    reverse_label_map = {0: 'foll', 1: 'fol', 2: 'agg', 3: 'uncertain'}
    y_pred_labels = [reverse_label_map[idx] for idx in y_pred_index]
    
    valid_mask = ~uncertain_mask
    if np.any(valid_mask):
        f1 = f1_score(y_test[valid_mask], y_pred_index[valid_mask], average='weighted')
        acc = accuracy_score(y_test[valid_mask], y_pred_index[valid_mask])
    else:
        f1 = 0.0
        acc = 0.0
        print("Warning: All test samples marked as uncertain")
    
    print("\nClassification Results:")
    print(f"- F1 Score (certain samples): {f1:.4f}")
    print(f"- Accuracy (certain samples): {acc:.4f}")
    print(f"- Uncertain samples: {np.mean(uncertain_mask):.2%}")
    
    results_df = pd.DataFrame({
        'Image_Path': [image_paths[i] for i in X_test.index],
        'True_Label': label_df.iloc[X_test.index],
        'Predicted_Label': y_pred_labels,
        'Prob_foll': y_proba[:, 0],
        'Prob_fol': y_proba[:, 1],
        'Prob_agg': y_proba[:, 2],
        'Max_Probability': max_proba,
        'Is_Uncertain': uncertain_mask
    })
    results_df.to_csv(os.path.join(output_path, 'classification_results.csv'), index=False)
    
    os.makedirs(WEIGHT_SAVE_PATH, exist_ok=True)
    model.save_model(os.path.join(WEIGHT_SAVE_PATH, 'tls_classifier.model'))
    np.save(os.path.join(WEIGHT_SAVE_PATH, 'scaler_params.npy'), 
            {'mean': scaler.mean_, 'scale': scaler.scale_})
    
    importance_df = pd.DataFrame({
        'Feature': feature_df.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(os.path.join(output_path, 'feature_importance.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))
    plt.close()
    
    if np.any(valid_mask):
        cm = confusion_matrix(
            y_test[valid_mask], 
            y_pred_index[valid_mask], 
            labels=[0, 1, 2]
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['foll', 'fol', 'agg'], 
                    yticklabels=['foll', 'fol', 'agg'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
        plt.close()
    
    print(f"\nFeature Importance Ranking:")
    print(importance_df)
    
    return model, f1, acc

if __name__ == "__main__":
    # USER CONFIGURATION
    INPUT_DATA_PATH = "path/to/training_data" 
    OUTPUT_RESULTS_PATH = "path/to/results"
    
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)
    
    print("="*50)
    print("TLS Classification System")
    print(f"Train/Test Split: {1-TEST_SIZE:.0f}:{TEST_SIZE:.0f}")
    print(f"Weight Save Path: {WEIGHT_SAVE_PATH}")
    print(f"Probability Threshold: {PROBABILITY_THRESHOLD}")
    print("="*50)
    
    model, f1, acc = train_tls_classifier(INPUT_DATA_PATH, OUTPUT_RESULTS_PATH)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Model saved to: {os.path.join(WEIGHT_SAVE_PATH, 'tls_classifier.model')}")
    print("="*50)