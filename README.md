# **Image Classification using CNN-FE and Transfer Learning**

This is a flask based version of our project. We changed the workflow to gradio for deploying it on HuggingFace. The gradio version is available at https://huggingface.co/spaces/dl-project-land-use/land-use-prediction?logs=container.

https://github.com/user-attachments/assets/26b6fd6b-b066-4d02-8420-3663ce52ff8a

## **Project Overview**

This project implements **two approaches** for image classification:

1. **CNN with Feature Extraction (CNN-FE)** - A custom-built Convolutional Neural Network model for feature extraction and classification.
2. **Transfer Learning & Fine-Tuning** - Leveraging a pre-trained **EfficientNet** model for feature extraction, followed by fine-tuning the last three fully connected layers.

Each method has been trained, evaluated, and compared based on **accuracy, training time, and generalization performance**.

---

## **Key Features**

- **CNN-FE Architecture**: Custom CNN with convolutional, batch normalization, and dropout layers.
- **Transfer Learning**: Uses **EfficientNet** as a feature extractor with only final layers trained.
- **Fine-Tuning**: Unfreezing and retraining the last **three fully connected layers** to improve accuracy.
- **Performance Tracking**: Training metrics, accuracy/loss graphs, and misclassified images are stored for analysis.
- **Model Selection**: Best-performing models are saved for testing and inference.

---

---

## **Setup Instructions**

### **1. Install Dependencies**

Ensure you have the required libraries installed:

```bash
pip install tensorflow numpy matplotlib pandas tqdm scikit-learn
```

### **2. Train CNN-FE Model**

Run the following notebook to train the CNN-FE model:

```bash
jupyter notebook cnn-fe/training-cnn-fe.ipynb
```

- Saves the best model as `AkhileshNegi-221AI008-model.h5`.
- Generates training and validation plots.

### **3. Train Transfer Learning Model**

Run the following notebook for Transfer Learning:

```bash
jupyter notebook Transfer_Learning/dl-transfer-learning.ipynb
```

- Uses **EfficientNet** as a feature extractor.
- Saves the best feature-extraction model as `efficientnetb7_transfer_learning.h5`.

### **4. Fine-Tune the Model**

To fine-tune the last three layers of **EfficientNet**, execute:

```bash
jupyter notebook Transfer-Learning/dl-transfer-learning_finetune.ipynb
```

- Saves the best fine-tuned model as `fine_tune_efficientnetb7_transfer_learning`.

### **5. Test the Model**

Run the testing notebooks for evaluation:

```bash
jupyter notebook cnn-fe/testing-cnn-fe.ipynb
jupyter notebook Transfer-Learning/dl-testing-transfer-learning.ipynb
jupyter notebook Transfer-Learning/dl-testing-transfer-learning_finetune.ipynb
```

---

## **Model Architectures**

### **CNN-FE Model**

- **Four convolutional layers** (with ReLU activation, Batch Normalization, and MaxPooling).
- **Fully connected layers** (512 and 256 neurons with dropout).
- **Softmax output layer** (multi-class classification).
- **Adam optimizer** and **Categorical Crossentropy loss function**.

### **Transfer Learning Model**

- **Base Model:** EfficientNetb7 with pre-trained weights (ImageNet).
- **Feature Extraction:** Only the final dense layers are trained.
- **Fine-Tuning:** Last **three fully connected layers** are unfrozen and trained.
- **Dropout** for regularization.

---

## **Results & Observations**

| Model             | Training Accuracy | Validation Accuracy | Testing Accuracy |
| ----------------- | ----------------- | ------------------- | ---------------- |
| CNN-FE            | **86.96%**        | 84.64%              | **79.29%**       |
| Transfer Learning | **91.63%**        | 91.43%              | **91.19%**       |
| Fine-Tuned TL     | **83.13%**        | 88.57%              | **86.94%**       |

- **TL performs best**, indicating that feature extraction capture more relevant patterns.
- **CNN-FE struggles with overfitting**, but batch normalization stabilizes activation distributions.
- **Transfer Learning generalizes well** even with a small dataset due to pre-trained EfficientNet features.

---

## **Contributors**

- **Student Name:** `Adya N A`
- **Reg. Number:** `221AI006`
- **Student Name:** `Akhilesh Negi`
- **Reg. Number:** `221AI008`

---

## **License**

This project is for educational purposes. Feel free to modify and experiment with the models.

---

Since saved models are large, these can be found in drive link: https://drive.google.com/drive/folders/1IearqwxcP6wR8mXPaUZtz3Ot-0weBviy?usp=sharing
