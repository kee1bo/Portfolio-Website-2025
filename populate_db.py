#!/usr/bin/env python3
"""
Script to populate the portfolio database with sample content.
Run this script to add sample blog posts, YouTube videos, and GitHub repositories.
"""

import sqlite3
from datetime import datetime, timedelta
import random

def populate_blog_posts():
    """Add sample blog posts to demonstrate the blog functionality."""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    sample_posts = [
        {
            'title': 'Building Scalable ML Pipelines with Apache Airflow',
            'slug': 'building-scalable-ml-pipelines-airflow',
            'content': '''# Building Scalable ML Pipelines with Apache Airflow

In the world of machine learning, moving from proof-of-concept to production-ready systems is one of the biggest challenges data scientists face. Apache Airflow has emerged as a powerful solution for orchestrating complex ML workflows.

## Why Airflow for ML Pipelines?

Apache Airflow provides several key advantages for ML pipeline orchestration:

- **Dependency Management**: Clear visualization of task dependencies
- **Retry Logic**: Automatic retry mechanisms for failed tasks
- **Monitoring**: Real-time monitoring and alerting capabilities
- **Scalability**: Easy horizontal scaling across multiple workers

## Core Components

### DAGs (Directed Acyclic Graphs)
DAGs are the heart of Airflow, defining the workflow structure and dependencies between tasks.

### Operators
Operators define what actually gets executed. For ML pipelines, we commonly use:
- `PythonOperator` for data processing
- `DockerOperator` for containerized model training
- `S3TransferOperator` for data movement

## Implementation Example

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline',
    schedule_interval='@daily',
    catchup=False
)

def extract_data():
    # Data extraction logic
    pass

def preprocess_data():
    # Data preprocessing logic
    pass

def train_model():
    # Model training logic
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

extract_task >> preprocess_task >> train_task
```

## Best Practices

1. **Idempotency**: Ensure tasks can be run multiple times safely
2. **Data Validation**: Include validation steps between major pipeline stages
3. **Resource Management**: Use appropriate pool configurations for resource-intensive tasks
4. **Error Handling**: Implement comprehensive error handling and alerting

## Monitoring and Alerting

Airflow's web UI provides excellent visibility into pipeline execution, but for production systems, consider integrating with:
- Prometheus for metrics collection
- Grafana for visualization
- Slack or email for alerting

## Conclusion

Apache Airflow transforms chaotic ML workflows into well-orchestrated, maintainable pipelines. By leveraging its powerful features, teams can build robust production ML systems that scale with their needs.

The investment in learning Airflow pays dividends in reduced operational overhead and increased confidence in ML deployments.''',
            'excerpt': 'Learn how to build robust, scalable machine learning pipelines using Apache Airflow for production ML systems.',
            'featured_image': 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800',
            'tags': 'Machine Learning,Apache Airflow,MLOps,Data Engineering,Python',
            'published': 1,
            'views': random.randint(150, 500),
            'reading_time': 8
        },
        {
            'title': 'Deep Dive into Transformer Architectures',
            'slug': 'deep-dive-transformer-architectures',
            'content': '''# Deep Dive into Transformer Architectures

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized natural language processing and has since found applications across various domains.

## The Attention Mechanism

At the heart of transformers lies the attention mechanism, which allows models to focus on relevant parts of the input sequence when processing each element.

### Self-Attention

Self-attention computes attention weights between all pairs of positions in a sequence:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attended_values)
```

## Position Encoding

Since transformers don't have inherent notion of sequence order, position encoding is crucial:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## Applications Beyond NLP

Transformers have found success in:
- **Computer Vision**: Vision Transformer (ViT)
- **Music Generation**: Music Transformer
- **Protein Folding**: AlphaFold2
- **Code Generation**: GitHub Copilot

## Training Considerations

### Optimization
- Use learning rate warmup
- Apply gradient clipping
- Consider mixed precision training

### Regularization
- Dropout in attention and feed-forward layers
- Layer normalization for stable training

## Future Directions

The transformer architecture continues to evolve with innovations like:
- Sparse attention patterns
- Retrieval-augmented generation
- Multi-modal transformers

Understanding transformers is essential for anyone working in modern AI, as they form the backbone of most state-of-the-art models.''',
            'excerpt': 'Explore the inner workings of transformer architectures that power modern AI systems like GPT and BERT.',
            'featured_image': 'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800',
            'tags': 'Deep Learning,Transformers,NLP,Neural Networks,AI',
            'published': 1,
            'views': random.randint(200, 800),
            'reading_time': 12
        },
        {
            'title': 'Implementing MLOps with Kubernetes and Kubeflow',
            'slug': 'implementing-mlops-kubernetes-kubeflow',
            'content': '''# Implementing MLOps with Kubernetes and Kubeflow

As machine learning models become increasingly complex and critical to business operations, the need for robust MLOps practices has never been greater. Kubernetes and Kubeflow provide a powerful platform for deploying and managing ML workflows at scale.

## What is MLOps?

MLOps (Machine Learning Operations) is the practice of collaboration and communication between data scientists and operations professionals to help manage the production ML lifecycle.

Key MLOps principles include:
- **Automation**: Automate the ML pipeline from data ingestion to model deployment
- **Monitoring**: Continuous monitoring of model performance and data drift
- **Versioning**: Track versions of data, code, and models
- **Reproducibility**: Ensure experiments and deployments are reproducible

## Kubernetes for ML

Kubernetes provides several advantages for ML workloads:

### Resource Management
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: ml-training
    image: tensorflow/tensorflow:2.8.0-gpu
    resources:
      requests:
        nvidia.com/gpu: 1
        memory: "8Gi"
        cpu: "4"
      limits:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "8"
```

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Kubeflow Pipelines

Kubeflow Pipelines enable you to build and deploy portable, scalable ML workflows:

```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

def preprocess_data(input_path: str, output_path: str):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Load and preprocess data
    df = pd.read_csv(input_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    # Save processed data
    processed_df = pd.DataFrame(scaled_data, columns=df.select_dtypes(include=[np.number]).columns)
    processed_df.to_csv(output_path, index=False)

def train_model(data_path: str, model_path: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Load data and train model
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, model_path)

# Create pipeline components
preprocess_op = create_component_from_func(preprocess_data)
train_op = create_component_from_func(train_model)

@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML training pipeline'
)
def ml_pipeline(input_data_path: str):
    # Data preprocessing step
    preprocess_task = preprocess_op(
        input_path=input_data_path,
        output_path='/tmp/processed_data.csv'
    )
    
    # Model training step
    train_task = train_op(
        data_path=preprocess_task.outputs['output_path'],
        model_path='/tmp/trained_model.pkl'
    )

# Compile and run pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ml_pipeline, 'ml_pipeline.yaml')
```

## Model Serving with KServe

KServe provides serverless inferencing for ML models:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
      resources:
        requests:
          cpu: 100m
          memory: 256Mi
        limits:
          cpu: 1
          memory: 1Gi
```

## Monitoring and Observability

### Prometheus Integration
```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('ml_predictions_total', 'Total ML predictions')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'ML prediction latency')

@prediction_latency.time()
def predict(model, input_data):
    prediction_counter.inc()
    return model.predict(input_data)
```

### Model Drift Detection
```python
import numpy as np
from scipy import stats

def detect_drift(reference_data, current_data, threshold=0.05):
    # Kolmogorov-Smirnov test for drift detection
    ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    if p_value < threshold:
        return True, f"Drift detected: p-value = {p_value}"
    return False, f"No drift detected: p-value = {p_value}"
```

## Best Practices

1. **Infrastructure as Code**: Use Helm charts or Kustomize for deployment configurations
2. **Security**: Implement RBAC and network policies
3. **Cost Optimization**: Use spot instances for training workloads
4. **Data Management**: Implement proper data versioning with DVC or similar tools

## Conclusion

Kubernetes and Kubeflow provide a robust foundation for implementing MLOps at scale. By leveraging these technologies, organizations can build reliable, scalable ML systems that deliver consistent business value.

The key is to start simple and gradually add complexity as your MLOps maturity grows.''',
            'excerpt': 'Learn how to implement scalable MLOps practices using Kubernetes and Kubeflow for production ML systems.',
            'featured_image': 'https://images.unsplash.com/photo-1667372393086-9d4001d51cf1?w=800',
            'tags': 'MLOps,Kubernetes,Kubeflow,DevOps,Machine Learning',
            'published': 1,
            'views': random.randint(100, 400),
            'reading_time': 15
        },
        {
            'title': 'Computer Vision with OpenCV and Deep Learning',
            'slug': 'computer-vision-opencv-deep-learning',
            'content': '''# Computer Vision with OpenCV and Deep Learning

Computer vision has experienced remarkable growth in recent years, driven by advances in deep learning and the availability of powerful libraries like OpenCV. This comprehensive guide explores modern computer vision techniques.

## Getting Started with OpenCV

OpenCV (Open Source Computer Vision Library) is one of the most popular libraries for computer vision tasks:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display an image
image = cv2.imread('sample.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
```

## Image Preprocessing

### Filtering and Enhancement
```python
# Gaussian blur for noise reduction
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Edge detection with Canny
edges = cv2.Canny(image, 50, 150)

# Histogram equalization for contrast enhancement
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

### Morphological Operations
```python
# Create kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

## Object Detection with YOLO

```python
import torch
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

def detect_objects(image_path):
    # Run inference
    results = model(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[class_id]}: {confidence:.2f}', 
                       (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Detect objects in image
detected_image = detect_objects('input.jpg')
cv2.imwrite('detected_output.jpg', detected_image)
```

## Image Segmentation

### Semantic Segmentation with DeepLab
```python
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101

def semantic_segmentation(image):
    # Load pre-trained DeepLab model
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Post-processing
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    return output_predictions
```

## Feature Detection and Matching

### SIFT Features
```python
def detect_and_match_features(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return kp1, kp2, good_matches
```

## Real-time Video Processing

```python
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Use webcam
    model = YOLO('yolov8n.pt')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Display frame
        cv2.imshow('Real-time Detection', annotated_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Custom CNN for Image Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')
```

## Performance Optimization

### GPU Acceleration
```python
# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move data to GPU
inputs = inputs.to(device)
labels = labels.to(device)
```

### Model Optimization
```python
# Model quantization for faster inference
import torch.quantization as quantization

# Post-training quantization
model_quantized = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

## Applications and Use Cases

1. **Medical Imaging**: Detecting tumors in MRI scans
2. **Autonomous Vehicles**: Object detection and lane recognition
3. **Manufacturing**: Quality control and defect detection
4. **Retail**: Product recognition and inventory management
5. **Security**: Face recognition and surveillance systems

## Best Practices

1. **Data Augmentation**: Increase dataset diversity
2. **Transfer Learning**: Leverage pre-trained models
3. **Proper Validation**: Use cross-validation for model evaluation
4. **Error Analysis**: Understand model failures and edge cases

## Conclusion

Computer vision with OpenCV and deep learning opens up endless possibilities for solving real-world problems. The combination of traditional computer vision techniques with modern deep learning approaches provides powerful tools for building intelligent visual systems.

As the field continues to evolve, staying updated with the latest techniques and best practices is essential for building effective computer vision applications.''',
            'excerpt': 'Comprehensive guide to modern computer vision techniques using OpenCV and deep learning frameworks.',
            'featured_image': 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800',
            'tags': 'Computer Vision,OpenCV,Deep Learning,YOLO,CNN',
            'published': 1,
            'views': random.randint(250, 600),
            'reading_time': 18
        },
        {
            'title': 'Getting Started with Large Language Models',
            'slug': 'getting-started-large-language-models',
            'content': '''# Getting Started with Large Language Models

Large Language Models (LLMs) have revolutionized natural language processing and opened up new possibilities for AI applications. This guide provides a comprehensive introduction to understanding and working with LLMs.

## What are Large Language Models?

LLMs are neural networks with billions of parameters trained on vast amounts of text data. They can understand and generate human-like text for various tasks including:

- Text generation and completion
- Question answering
- Summarization
- Translation
- Code generation
- Creative writing

## Popular LLM Architectures

### GPT (Generative Pre-trained Transformer)
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "The future of artificial intelligence is"
generated_text = generate_text(prompt)
print(generated_text)
```

### BERT for Understanding Tasks
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model for classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return predictions.cpu().numpy()

# Example usage
text = "I love this new technology!"
sentiment_scores = classify_sentiment(text)
```

## Fine-tuning LLMs

### Dataset Preparation
```python
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load your dataset
df = pd.read_csv('your_dataset.csv')
dataset = CustomDataset(df['text'].values, df['label'].values, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Training Loop
```python
from transformers import AdamW
from torch.nn import CrossEntropyLoss

def train_model(model, dataloader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

# Fine-tune the model
train_model(model, dataloader)
```

## Working with APIs

### OpenAI API
```python
import openai

openai.api_key = 'your-api-key'

def chat_with_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

# Example conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = chat_with_gpt(messages)
print(response)
```

### Hugging Face Inference API
```python
import requests

def query_huggingface_model(text, model_name="gpt2"):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    return response.json()

# Generate text using Hugging Face API
result = query_huggingface_model("The benefits of machine learning include")
print(result)
```

## Prompt Engineering

### Effective Prompting Strategies
```python
def create_few_shot_prompt(task_description, examples, query):
    prompt = f"{task_description}\n\n"
    
    for example in examples:
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    prompt += f"Input: {query}\nOutput:"
    return prompt

# Example: Sentiment analysis
task_desc = "Classify the sentiment of the following text as positive, negative, or neutral."
examples = [
    {"input": "I love this product!", "output": "positive"},
    {"input": "This is terrible.", "output": "negative"},
    {"input": "It's okay, I guess.", "output": "neutral"}
]

query = "The movie was amazing!"
prompt = create_few_shot_prompt(task_desc, examples, query)
```

### Chain of Thought Prompting
```python
def chain_of_thought_prompt(problem):
    prompt = f"""
    Problem: {problem}
    
    Let's think step by step:
    1. First, I need to understand what the problem is asking
    2. Then, I'll identify the key information
    3. Next, I'll work through the solution methodically
    4. Finally, I'll provide the answer
    
    Solution:
    """
    return prompt

# Example usage
math_problem = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"
prompt = chain_of_thought_prompt(math_problem)
```

## Model Evaluation

### Perplexity for Language Models
```python
import torch
import math

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
```

### BLEU Score for Generation
```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate_generation(reference, hypothesis):
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens)
    return bleu_score

# Example evaluation
reference = "The cat is sitting on the mat"
generated = "A cat is sitting on a mat"
score = evaluate_generation(reference, generated)
print(f"BLEU Score: {score:.4f}")
```

## Deployment Considerations

### Model Optimization
```python
# Model quantization for faster inference
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Compare model sizes
original_size = sum(p.numel() for p in model.parameters())
quantized_size = sum(p.numel() for p in quantized_model.parameters())

print(f"Original model parameters: {original_size:,}")
print(f"Quantized model parameters: {quantized_size:,}")
```

### Caching and Batch Processing
```python
from functools import lru_cache
import asyncio

@lru_cache(maxsize=1000)
def cached_generation(prompt, max_length=50):
    """Cache frequently used generations"""
    return generate_text(prompt, max_length)

async def batch_generate(prompts, batch_size=8):
    """Process multiple prompts in batches"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            asyncio.to_thread(generate_text, prompt) 
            for prompt in batch
        ])
        results.extend(batch_results)
    
    return results
```

## Ethical Considerations

When working with LLMs, consider:

1. **Bias Mitigation**: Regularly audit model outputs for biases
2. **Content Filtering**: Implement safeguards against harmful content
3. **Privacy**: Ensure user data is handled responsibly
4. **Transparency**: Be clear about AI-generated content
5. **Environmental Impact**: Consider the carbon footprint of large models

## Future Trends

- **Multimodal Models**: Integration of text, image, and audio
- **Efficient Architectures**: Smaller models with comparable performance
- **Specialized Models**: Domain-specific LLMs for medicine, law, etc.
- **Interactive AI**: More sophisticated conversational agents

## Conclusion

Large Language Models represent a paradigm shift in AI capabilities. Understanding how to effectively work with these models—from fine-tuning to deployment—is crucial for building next-generation AI applications.

As the field continues to evolve rapidly, staying updated with best practices and emerging techniques will be essential for leveraging the full potential of LLMs.''',
            'excerpt': 'Complete guide to understanding, fine-tuning, and deploying Large Language Models for various applications.',
            'featured_image': 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800',
            'tags': 'LLM,GPT,BERT,NLP,Machine Learning',
            'published': 1,
            'views': random.randint(300, 900),
            'reading_time': 20
        }
    ]
    
    for post in sample_posts:
        # Check if post already exists
        c.execute("SELECT id FROM blog_posts WHERE slug = ?", (post['slug'],))
        if c.fetchone() is None:
            # Create timestamp
            created_at = datetime.now() - timedelta(days=random.randint(1, 90))
            
            c.execute("""
                INSERT INTO blog_posts (title, slug, content, excerpt, featured_image, tags, published, created_at, views, reading_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post['title'], post['slug'], post['content'], post['excerpt'],
                post['featured_image'], post['tags'], post['published'],
                created_at.isoformat(), post['views'], post['reading_time']
            ))
    
    conn.commit()
    conn.close()
    print(f"Added {len(sample_posts)} blog posts to the database.")

def populate_youtube_videos():
    """Add sample YouTube videos."""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    sample_videos = [
        {
            'title': 'Building Scalable ML Pipelines',
            'video_id': 'dQw4w9WgXcQ',
            'description': 'Learn how to build production-ready machine learning pipelines',
            'thumbnail_url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'published_at': datetime.now() - timedelta(days=30),
            'views': 15420,
            'duration': '12:34',
            'featured': 1
        },
        {
            'title': 'Deep Learning for NLP',
            'video_id': 'dQw4w9WgXcQ',
            'description': 'Comprehensive guide to neural networks for natural language processing',
            'thumbnail_url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'published_at': datetime.now() - timedelta(days=60),
            'views': 8930,
            'duration': '18:45',
            'featured': 1
        },
        {
            'title': 'Computer Vision with PyTorch',
            'video_id': 'dQw4w9WgXcQ',
            'description': 'Hands-on tutorial for computer vision using PyTorch',
            'thumbnail_url': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'published_at': datetime.now() - timedelta(days=90),
            'views': 12340,
            'duration': '22:15',
            'featured': 0
        }
    ]
    
    for video in sample_videos:
        c.execute("SELECT id FROM youtube_videos WHERE video_id = ?", (video['video_id'],))
        if c.fetchone() is None:
            c.execute("""
                INSERT INTO youtube_videos (title, video_id, description, thumbnail_url, published_at, views, duration, featured)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video['title'], video['video_id'], video['description'],
                video['thumbnail_url'], video['published_at'].isoformat(),
                video['views'], video['duration'], video['featured']
            ))
    
    conn.commit()
    conn.close()
    print(f"Added {len(sample_videos)} YouTube videos to the database.")

def populate_github_repos():
    """Add sample GitHub repositories."""
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    sample_repos = [
        {
            'name': 'sentiment-analysis-engine',
            'description': 'Multi-lingual sentiment analysis using transformer architectures',
            'html_url': 'https://github.com/kee1bo/sentiment-analysis',
            'language': 'Python',
            'stars': 156,
            'forks': 23,
            'updated_at': datetime.now() - timedelta(days=15),
            'featured': 1
        },
        {
            'name': 'computer-vision-toolkit',
            'description': 'Comprehensive computer vision tools and utilities',
            'html_url': 'https://github.com/kee1bo/computer-vision',
            'language': 'Python',
            'stars': 89,
            'forks': 12,
            'updated_at': datetime.now() - timedelta(days=30),
            'featured': 1
        },
        {
            'name': 'ml-pipeline-framework',
            'description': 'Scalable machine learning pipeline framework with Airflow',
            'html_url': 'https://github.com/kee1bo/ml-pipelines',
            'language': 'Python',
            'stars': 234,
            'forks': 45,
            'updated_at': datetime.now() - timedelta(days=7),
            'featured': 1
        }
    ]
    
    for repo in sample_repos:
        c.execute("SELECT id FROM github_repos WHERE name = ?", (repo['name'],))
        if c.fetchone() is None:
            c.execute("""
                INSERT INTO github_repos (name, description, html_url, language, stars, forks, updated_at, featured)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                repo['name'], repo['description'], repo['html_url'],
                repo['language'], repo['stars'], repo['forks'],
                repo['updated_at'].isoformat(), repo['featured']
            ))
    
    conn.commit()
    conn.close()
    print(f"Added {len(sample_repos)} GitHub repositories to the database.")

if __name__ == '__main__':
    print("Populating portfolio database with sample data...")
    
    populate_blog_posts()
    populate_youtube_videos()
    populate_github_repos()
    
    print("\nDatabase population completed!")
    print("You can now run the Flask application to see the enhanced portfolio website.")
    print("Run: python app.py")