# Sentiment-Analysis-With-BERT

BERT is super powerul method to perform sentiment analysis. This project was built by using three different datasets:

- Women’s E-Commerce Clothing Reviews
- Amazon Fine Food Reviews
- TripAdvisor Hotel Reviews

The system's main steps was shown below as a diagram.

![steps](https://user-images.githubusercontent.com/57035819/150503632-88def8c6-cd41-4a2d-afbb-ec5a3a8cf666.jpg)

Each dataset were prepared to process and this includes the following steps:

- Reading and visualizing data to analyse distribution
- Data preprocessing
  - Dropping NA values
  - Lower case
  - Punctuations removal
  - Whitespaces removal
  - Stopwords removal
  - Lemmatization
- Building Machine Learning models and fine-tuning
- Using BERT Base Multilingual Uncased Sentiment for prediction
- Analysis of success of all methods
- Finding the best and worst sectors in the datasets

Machine learning models did not require huge effort to perform the task. After preparation of the datasets with appropriate steps, the datasets were ready to process. The
following machine learning algorithms were used in the project:

- Logistic Regression
- Support Vectore Machines (SVM)
- K-Nearest Neighbour
- Multi-Layer Perceptron

Due to the imbalance of datasets (generally comments with 4 and 5 star ratings were in high number in data samples) some methods were implemented to create new data samples to
overcome this imbalance case. The methods were listed below:

- Random Over Sampling
- Synthetic Minority Over Sampling Technique
- Near Miss Undersampling

Each method has their own way of creating new data samples.

BERT Base Multilingual Uncased Sentiment pre-trained model were implemented to achieve the goal of the project. This model works efficiently and produces star rating value
for given text input which is a comment & evaluation about something. Some examples of this model and outputs were showed in the next figure.

![bert_bmus](https://user-images.githubusercontent.com/57035819/150504716-0fb9df69-41f3-46df-bc9c-9a48ec26837f.png)

## Author
- [honourrable](https://github.com/honourrable)
