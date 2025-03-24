# Automated Lesion Detection in Small Intestine Capsule Endoscopy Videos

## **Project Overview**  
This project aims to develop a deep learning pipeline for **automatic lesion detection in capsule endoscopy videos**. The methodology focuses on reducing redundancy while ensuring comprehensive lesion detection by combining **CNN-based feature extraction**, **selective frame sampling**, and **temporal modeling with LSTM/GRU**.  

## **Motivation**  
Capsule endoscopy generates hours of video data, making manual review tedious and error-prone. Existing techniques either miss subtle lesions or produce redundant detections due to similar consecutive frames. Our solution optimizes lesion detection by leveraging temporal information, thereby enhancing accuracy and reducing redundant findings.  

## **Planned Methodology**  

1. **Data Preprocessing:**  
   - Extract frames from videos at regular intervals.  
   - Apply augmentations like rotation, brightness adjustment, and flipping to improve model robustness.  

2. **Feature Extraction:**  
   - Use a **CNN (e.g., ResNet, EfficientNet)** to extract embeddings from each frame.  
   - Optionally fine-tune the CNN on the dataset for domain adaptation.  

3. **Selective Frame Sampling:**  
   - Implement strategies to reduce redundancy:  
     - **Cosine Similarity / Euclidean Distance**: Compare embeddings of consecutive frames to select only distinct ones.  
     - **Threshold-based Sampling**: Drop frames where the similarity is above a set threshold.  
     - **Non-Maximum Suppression (NMS)**: Suppress redundant detections by retaining only the most confident predictions.  

4. **Temporal Modeling:**  
   - Pass the sampled frame embeddings into an **LSTM/GRU** to model temporal dependencies.  
   - The LSTM/GRU outputs:  
     - **Frame-level predictions**: Detect lesions at each frame.  
     - **Video-level predictions**: Aggregate predictions across the video for overall lesion detection.  

5. **Redundancy Removal and Timestamping:**  
   - Apply post-processing to:  
     - Merge consecutive detections of the same lesion.  
     - Assign **timestamps** to the detected lesions for easy video navigation.  

6. **Evaluation Metrics:**  
   - **Frame-level metrics**: Precision, Recall, F1-score, Accuracy.  
   - **Video-level metrics**: Lesion detection rate, Redundant detection rate.  
   - **Temporal metrics**: Mean time error (difference between predicted and actual lesion timestamps).  

7. **Addressing Class Imbalance:**  
   - Techniques like **Oversampling of minority classes**, **Focal Loss**, and **Class Weighting** will be explored to handle imbalance in lesion types.  

## **Datasets:**  
- **Kvasir Capsule Dataset** — Contains labeled endoscopy videos across multiple lesion types. Labels are frame-wise and identify specific lesion types.  
- **SeeAI Dataset** — Provides additional labeled capsule endoscopy videos, enabling better generalization and robustness of the model.  

## **Tools and Technologies:**  
- Python, TensorFlow/Keras, OpenCV, NumPy, Scikit-learn.  
- CNN Architectures: ResNet, EfficientNet.  
- Sequential Models: LSTM, GRU.  

## **Timeline:**  
1. Data Preprocessing and CNN Training (1 week).  
2. Selective Frame Sampling Implementation (1 week).  
3. Temporal Modeling and Integration (2 weeks).  
4. Evaluation and Optimization (1 week).  

## **Future Scope:**  
- Integration with a **real-time video analysis system**.  
- Exploration of object tracking techniques to enhance redundancy removal.  
- Extending the pipeline to support multi-lesion tracking.  

---
