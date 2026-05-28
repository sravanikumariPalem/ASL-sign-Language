# Sign Language Recognition using Deep Learning
 
This project is part of my personal learning portfolio in deep learning and computer vision. I built it to get hands-on experience with real classification problems that go beyond toy datasets, and to understand how different model architectures behave on the same task. It is not a research paper and it is not a finished product. It is an honest record of what I built, what I learned, and where I want to take it further.
 
---
 
## What this project is about
 
American Sign Language (ASL) has 26 hand gestures for letters and 10 for digits, making it a 36-class image classification problem. What makes it genuinely interesting as a learning project is that some signs look nearly identical in a single photograph. The difference between certain letters comes down to a slight finger angle or a movement you cannot see in one frozen frame. That pushed me to go beyond a simple CNN and think about how to bring temporal context into the pipeline, which led to the three-model approach I ended up with.
 
---
 
## Why three models
 
I did not set out to build three models. I started with the CNN, got it working, and then ran into the problem I just described: visually similar signs were being confused with each other. That led me to look into sequence modelling, which introduced the CNN-LSTM combination. Once I had embeddings coming out of the LSTM, I wanted to understand them better, and swapping the KNN for a Random Forest gave me feature importance scores that let me actually inspect the embedding space. Each model grew out of a question the previous one raised.
 
---
 
## Models
 
### Standalone CNN
 
A four-block convolutional network that classifies each image on its own. I used Batch Normalisation after each convolutional layer and Global Average Pooling before the dense head instead of Flatten, which reduced the parameter count and helped with generalisation. The learning rate is set at 3e-4 with a ReduceLROnPlateau scheduler, and Early Stopping prevents the model from training past the point of improvement.
 
One thing I learned building this: the ordering of BatchNorm and activation matters more than I expected. Placing BatchNorm before the activation (Conv -> BN -> ReLU) gives more stable training than the other way around, particularly in the early epochs when the running statistics are still settling.
 
```
Conv(32) -> BN -> ReLU -> MaxPool
Conv(64) -> BN -> ReLU -> MaxPool
Conv(128) -> BN -> ReLU -> MaxPool
Conv(256) -> ReLU -> GlobalAveragePooling
Dense(256) -> BN -> ReLU -> Dropout(0.4)
Dense(128) -> ReLU -> Dropout(0.3)
Dense(36, softmax)
```
 
### CNN-LSTM with KNN
 
Instead of classifying a single image, this model builds a sequence of five frames per sample and processes each frame through a shared CNN backbone using TimeDistributed. The frame-level feature vectors go into a two-layer stacked LSTM, and the final LSTM output is used as a fixed-length embedding for the whole sequence. A KNN classifier is then trained on those embeddings.
 
Separating the feature extractor from the classifier like this was something I wanted to try specifically because it makes the components independently modifiable. The CNN-LSTM can be improved without touching the classifier, and the classifier can be swapped without retraining the deep learning component.
 
For the KNN I used cosine distance rather than Euclidean. I tested both and cosine worked noticeably better, which makes sense because in high-dimensional embedding spaces the magnitude of vectors matters less than their direction. I also ran a sweep across k values from 1 to 20 and picked the best performing k rather than guessing.
 
```
Input: (5, 64, 64, 3) sequences
TimeDistributed(CNN backbone) -> per-frame embeddings
LSTM(128, return_sequences=True)
LSTM(64, return_sequences=False) -> 64-dim embedding
KNN(k=5, metric=cosine, weights=distance)
```
 
### CNN-LSTM with Random Forest
 
The same CNN-LSTM extractor as above, but with a Random Forest classifier on the embeddings instead of KNN. My reason for trying this was interpretability. KNN gives you a prediction but tells you nothing about the structure of the embedding space. Random Forest gives you feature importance scores, which let you see which of the 64 embedding dimensions are actually carrying the discriminative signal and which are noise.
 
The forest uses 300 trees with square root feature sampling and balanced class weights. It trains across all available CPU cores so it is reasonably fast even at 300 estimators.
 
```
Same CNN-LSTM extractor as above
RandomForest(n_estimators=300, max_features=sqrt, class_weight=balanced)
```
 
---
 
## Dataset
 
The project uses the ASL dataset which contains labelled hand gesture images across all 36 classes. Images are resized to 64x64 for training speed. Increasing this to 224x224 would improve accuracy but takes significantly longer to train on a CPU.


download dataset from https://www.kaggle.com/datasets/ayuraj/asl-dataset
 
Update the `DATASET_PATH` variable at the top of the notebook before running.
 
---
 
## Setup
 
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn pillow joblib
```
 
Then open `signlan.ipynb` in Jupyter and run the cells in order. All three models train sequentially. Models 2 and 3 share the same CNN-LSTM extractor so the feature extraction only runs once.
 
---
 
## Configuration
 
```python
IMG_H = IMG_W  = 64       # resolution -- increase to 224 for better accuracy
SEQ_LEN        = 5        # frames per sequence
SEQS_PER_CLASS = 50       # sequences sampled per class
TEST_FRAC      = 0.20
VALID_FRAC     = 0.10
BATCH          = 32
EPOCHS         = 50
SEED           = 7
```
 
---
 
## Saved files
 
After training, the following are written to the working directory:
 
```
asl_cnn.h5                   standalone CNN weights
asl_cnn_lstm_extractor.h5    CNN-LSTM extractor
asl_knn.pkl                  KNN classifier
asl_rf.pkl                   Random Forest classifier
asl_label_enc_cnn.pkl        label encoder for CNN
asl_label_enc_seq.pkl        label encoder for sequence models
cnn_best.keras               best CNN checkpoint
```
 
The notebook includes `predict_cnn()` and `predict_rf()` helper functions that load these files and run inference on new images without re-running any training.
 
---
 
## What I learned
 
The most useful thing I took from this project was not any particular result but the process of diagnosing why a model was not learning. The CNN went through a phase where accuracy was stuck at roughly 3%, which for 36 classes is essentially random. Tracking that down to a combination of BatchNorm ordering, an overly aggressive learning rate reducer, and feeding augmented batches through a BatchNorm layer before the statistics had stabilised took me longer than I expected, but I came out of it with a much clearer mental model of how these components interact.
 
The sequence modelling work introduced me to TimeDistributed wrappers, stacked LSTMs, and the idea of using a deep learning model purely as a feature extractor rather than as an end-to-end classifier. That last point feels like it will be useful well beyond this project.
 
The Random Forest addition was genuinely satisfying because it turned an opaque embedding into something I could inspect. Seeing which dimensions of the LSTM output carry the most weight for classification made the whole pipeline feel less like a black box.
 
---
 
## What I would do differently or next
 
The sequences in Models 2 and 3 are built by randomly sampling multiple images from the same class, not from real video. It is a reasonable stand-in for learning purposes but it does not capture how a sign actually moves through time. Real video data would be a meaningful upgrade.
 
The project covers ASL. British Sign Language uses a two-handed alphabet and has a different structure entirely. Extending this to BSL is something I want to work on, partly because it is the more practically relevant target for a UK context and partly because the two-handed nature of the signs would make the computer vision problem more interesting.
 
At 64x64, the model is learning from fairly low-resolution representations. Training at 224x224 is the obvious next step if compute allows.
 
---
 
## Tools and libraries
 
Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Pillow, joblib
