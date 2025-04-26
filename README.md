# Automated Lesion Detection in Small Intestine Capsule Endoscopy Videos


## Table of Contents

-   [Introduction](#introduction)
-   [Problem Statement](#problem-statement)
-   [Project Goal](#project-goal)
-   [Features](#features)
-   [Methodology](#methodology)
-   [Datasets](#datasets)
-   [Technology Stack](#technology-stack)
-   [Results](#results)
-   [Limitations & Future Work](#limitations--future-work)
-   [Contributing](#contributing)
-   [Acknowledgements](#acknowledgements)

## Introduction

Capsule Endoscopy (CE) is a non-invasive medical procedure that utilizes a pill-sized camera to capture thousands of images while traversing the gastrointestinal (GI) tract, particularly the hard-to-reach small intestine. It is crucial for diagnosing various GI disorders like obscure bleeding, Crohn's disease, tumors, polyps, and infections.

## Problem Statement

Manual review of lengthy CE videos (often containing tens of thousands of frames) by gastroenterologists is extremely time-consuming (can take hours per video) and prone to human error, potentially leading to missed diagnoses, especially for subtle lesions.

## Project Goal

This project aims to develop and evaluate a deep learning-based system for the automated detection of lesions (such as ulcers, polyps, bleeding areas, angiectasia, etc.) in small intestine capsule endoscopy videos. The system intends to:

-   Improve diagnostic efficiency and accuracy.
-   Reduce the diagnostic time required by medical professionals.
-   Minimize the risk of oversight and missed lesions.
-   Provide consistent detection accuracy.

## Features

-   Utilizes state-of-the-art CNNs for spatial feature extraction from video frames.
-   Employs selective frame sampling techniques (Cosine Similarity, Euclidean Distance) to reduce redundancy and computational load.
-   Incorporates Recurrent Neural Networks (Bi-LSTM) to model temporal dependencies between frames.
-   Detects various types of lesions and anatomical landmarks.
-   Evaluated on diverse public datasets (Kvasir-Capsule, SEE-AI).
-   Provides frame-level and video-level evaluation metrics.
-   Visualizes detected lesions by overlaying predictions on video frames with timestamps.
-   Generates a summary report of detected lesions per video.

## Methodology

The automated lesion detection pipeline follows these steps:

1.  **Data Acquisition:** Utilizes the Kvasir-Capsule and SEE-AI datasets containing CE videos and annotated frames.
2.  **Data Preprocessing:**
    *   Extract frames from raw videos.
    *   Resize frames to a consistent size (e.g., 224x224).
    *   Normalize pixel values.
    *   Handle class imbalance using techniques like random weighted sampling.
3.  **Feature Extraction:** Pass each preprocessed frame through a pre-trained CNN backbone (e.g., ResNet, DenseNet, EfficientNet) to extract deep feature embeddings from the penultimate layer. DenseNet showed the best performance in initial tests.
4.  **Selective Frame Sampling (Redundancy Reduction):**
    *   Compare consecutive frame embeddings using Cosine Similarity or Euclidean Distance.
    *   Retain only frames that show significant change (similarity below or distance above a threshold), reducing the number of frames processed temporally.
5.  **Temporal Modeling:** Feed the sequence of selected frame embeddings into a Bidirectional LSTM (Bi-LSTM) network to capture temporal patterns and dependencies relevant to lesion detection across frames.
6.  **Classification/Detection:** A final layer predicts the presence and type of lesion (or normal tissue/landmark) for the input sequence elements.
7.  **Evaluation:** Assess performance using:
    *   **Frame-Level:** Accuracy, Precision, Recall, F1-Score.
    *   **Video-Level:** Lesion Detection Rate (LDR - did the model find *at least one* lesion in a positive video?), Redundancy Reduction Rate (RRR).
8.  **Post-Processing & Visualization:** Overlay bounding boxes or masks on video frames corresponding to detected lesions, annotate with timestamps, and generate summary reports. 

## Datasets

1.  **Kvasir-Capsule:** A large, public VCE dataset with 47,238 labeled frames across 14 classes (including lesions, landmarks, and normal views) and 117 full VCE videos (~4.7M frames).
    *   *Challenge:* Significant class imbalance.
2.  **SEE-AI:** A public dataset with 18,481 CE images from the small intestine and 23,033 annotations.

Training on diverse datasets helps ensure the model learns robust representations and generalizes better.

## Technology Stack

-   **Language:** Python 3.x
-   **Deep Learning:** PyTorch / TensorFlow / Keras (Specify which one was used)
-   **Computer Vision:** OpenCV (`opencv-python`)
-   **Machine Learning:** Scikit-learn (`sklearn`) for metrics and potentially KNN evaluation.
-   **Data Handling:** NumPy, Pandas
-   **Video Processing:** FFmpeg (may be needed for frame extraction)


## Results

The models were evaluated on their ability to detect lesions accurately and efficiently:

-   **Frame Sampling:**
    -   Cosine Similarity (threshold 0.98) achieved a Redundancy Reduction Rate (RRR) of **63.95%**, retaining only ~35% of frames.
    -   Euclidean Distance (threshold 5.0) achieved an RRR of **0.04%**, retaining almost all frames.
-   **Classification Performance (Bi-LSTM):**
    | Method              | Accuracy | Recall | F1-Score |
    | :------------------ | :------- | :----- | :------- |
    | Without Sampling    | 81.88%   | 0.8578 | 0.8255   |
    | Cosine Sampling     | 89.85%   | 0.8985 | 0.8870   |
    | Euclidean Sampling  | **90.84%**| **0.9084**| **0.8831**|
-   **Video-Level Detection:** The Lesion Detection Rate (LDR) was **1.0** for all trained models, indicating that at least one lesion was correctly identified in every ground truth video containing lesions.

*Summary:* The Bi-LSTM model combined with Euclidean distance-based frame sampling yielded the highest frame-level accuracy, while Cosine sampling significantly reduced the number of frames needed for analysis with a slight trade-off in accuracy compared to Euclidean sampling.

## Limitations & Future Work

Based on the literature review and project findings:

-   **Computational Requirements:** Deep learning models, especially hybrid ones, can be computationally intensive, potentially requiring specialized hardware (GPUs).
-   **Dataset Variability & Generalization:** Performance can vary across different datasets and patient populations. Further testing on more diverse data is needed.
-   **Model Interpretability:** Understanding *why* a model makes a certain prediction (explainability) is crucial for clinical trust and adoption. Integrating explainability techniques (e.g., attention maps, SHAP) is important future work.
-   **Rare Lesions:** Models might struggle with very rare lesion types due to limited training examples (Few-Shot Learning approaches like Lesion2Vec could be explored further).
-   **Real-time Performance:** Optimizing the pipeline for real-time or near-real-time processing in a clinical setting.

## Contributing

Contributions are welcome! You can contribute by:

-   Reporting bugs
-   Suggesting enhancements
-   Submitting pull requests with new features or fixes
-   Improving documentation


## Acknowledgements

This project is based on the research paper:

*   **Title:** Automated Lesion Detection in Small Intestine Capsule Endoscopy Videos
*   **Authors:** Rhea Chainani, Sakshi Sah, Suhani Thakur, Twsha Vyass, Dr. Nilkanth Deshpande, Dr. Shivali Wagle
*   **Institution:** Department of AI&ML, Symbiosis Institute of Technology, Pune, India
*   *(Add reference to the publication venue/preprint link if available)*

We thank the creators of the Kvasir-Capsule and SEE-AI datasets for making their valuable data publicly available.

