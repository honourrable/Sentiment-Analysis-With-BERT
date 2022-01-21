from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NearMiss
from nltk.stem import WordNetLemmatizer
from IPython.display import display
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from datetime import timedelta
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
import winsound
import string
import torch
import nltk
import time


bert_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
bert_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.5
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 16

counter = [0]
sample_number = [0]


def prepare_original_data():
    # Reading, preparing and viewing datasets
    df = pd.read_csv('datasets/clothes.csv', na_values=' ').dropna()
    selected_columns = df[["Review Text", "Rating"]]
    df_clothes = selected_columns.copy()
    df_clothes["Review Text"] = df_clothes["Review Text"].astype('string')

    df = pd.read_csv('datasets/dishes.csv', na_values=' ').dropna()
    df.drop(df.tail(544968).index, inplace=True)
    selected_columns = df[["Text", "Score"]]
    df_dishes = selected_columns.copy()
    df_dishes["Text"] = df_dishes["Text"].astype('string')

    df_hotels = pd.read_csv('datasets/hotels.csv', na_values=' ').dropna()
    df_hotels["Review"] = df_hotels["Review"].astype('string')

    display(df_clothes)
    display(df_dishes)
    display(df_hotels)

    # Preprocessing and saving new datasets
    print("\nBefore preprocessing:\n", df_clothes.iloc[15]["Review Text"])
    cached_stopwords = stopwords.words("english")
    df_clothes = preprocessing(df_clothes, cached_stopwords)
    df_dishes = preprocessing(df_dishes, cached_stopwords)
    df_hotels = preprocessing(df_hotels, cached_stopwords)
    print("\nAfter preprocessing:\n", df_clothes.iloc[15]["Review Text"])

    df_clothes.to_csv('datasets/preprocessed_clothes.csv')
    df_dishes.to_csv('datasets/preprocessed_dishes.csv')
    df_hotels.to_csv('datasets/preprocessed_hotels.csv')

    return


def preprocessing(dataframe, cached_stopword):
    wordnet_lemmatizer = WordNetLemmatizer()

    for index, row in dataframe.iterrows():
        text = row[dataframe.columns[0]]
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = ' '.join([word for word in text.split() if word not in cached_stopword])
        tokens = nltk.word_tokenize(text)
        text = ' '.join([wordnet_lemmatizer.lemmatize(token) for token in tokens])

        dataframe.at[index, dataframe.columns[0]] = text

    return dataframe


def example_bmus(text):
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)

    print(result.logits)
    print("Expected class:", int(torch.argmax(result.logits))+1)

    return


def bert_bmus(text):
    # BERT processes 512 tokens at most at a time
    tokens = bert_tokenizer.encode(text[:512], return_tensors='pt')
    result = bert_model(tokens)

    # All class predictions
    # print(result.logits)

    # These lines are written to show how many samples left to be processed
    counter[0] += 1
    if int(counter[0] % int(sample_number[0] / 5)) == 0:
        print_progress(counter[0], sample_number[0], length=50, printEnd='\r')

    return int(torch.argmax(result.logits)) + 1


def lr_classifier(cv_train, cv_test, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(cv_train, y_train)

    y_preds = model.predict(cv_test)

    return y_preds


def svm_classifier(cv_train, cv_test, y_train):
    model = SVC(kernel='linear', random_state=0)
    model.fit(cv_train, y_train)

    y_preds = model.predict(cv_test)

    return y_preds


def knn_classifier(cv_train, cv_test, y_train):
    model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    model.fit(cv_train, y_train)

    y_preds = model.predict(cv_test)

    return y_preds


def mlp_classifier(cv_train, cv_test, y_train):
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
    nn_classifier.fit(cv_train, y_train)

    y_preds = nn_classifier.predict(cv_test)

    return y_preds


def evaluate_preds(y_test, y_preds, method_name):
    conf_matrix = confusion_matrix(y_test, y_preds)

    print("\nConfussion Matrix:\n\n", conf_matrix, sep='')
    print("\nSuccess Metrics:\n", classification_report(y_test, y_preds))

    fig, axs = plt.subplots(ncols=2)

    sns.countplot(x=y_preds, ax=axs[0]).set(title=method_name)

    # Conversion of star ratings to 3 sentiment classes: Negative, Neutral, Positive
    convert_to_sentiment_v = np.vectorize(convert_to_sentiment)
    y_preds_converted = convert_to_sentiment_v(y_preds)
    y_test_converted = convert_to_sentiment_v(y_test)

    unique, counts = np.unique(y_preds_converted, return_counts=True)
    sentiment_analysis = dict(zip(unique, counts))
    print("Reduction to 3 classes:\n", sentiment_analysis)

    class_names = ['Negative', 'Neutral', 'Positive']
    g2 = sns.countplot(x=y_preds_converted, ax=axs[1])
    g2.set_xticklabels(class_names)
    plt.show()

    print("\nAfter Class Reduction:\n", classification_report(y_test_converted, y_preds_converted))

    return


def convert_to_sentiment(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def run_for_clothing_dataset(apply_bert_bmus):
    print("\n\n\t\t\tWomen's E-Commerce Clothing Reviews Star Rating Analysis:")
    print("\nLoaded preprocessed dataset:\n")


    # Reading Data
    clothes = pd.read_csv('datasets/preprocessed_clothes.csv', index_col=[0])
    clothes_text = clothes['Review Text'].to_numpy()
    clothes_labels = clothes['Rating'].to_numpy()


    # Data Visualization
    fig, axs = plt.subplots(ncols=4)
    display(clothes)
    print(clothes.info)

    X_ros, y_ros = ros_sampling(clothes_text, clothes_labels)
    X_smote, y_smote = smote_sampling(clothes_text, clothes_labels)
    X_nm, y_nm = near_miss_sampling(clothes_text, clothes_labels)

    sns.countplot(x=clothes['Rating'], ax=axs[0]).set(title='Original Data')
    sns.countplot(x=y_ros, ax=axs[1]).set(title='After ROS')
    sns.countplot(x=y_smote, ax=axs[2]).set(title='After SMOTE')
    sns.countplot(x=y_nm, ax=axs[3]).set(title='After NMS')
    plt.show()


    # Star Rating (Sentiment) Analysis
    X_train, X_test, y_train, y_test = train_test_split(clothes_text, clothes_labels, test_size=0.2)

    # Text to numerical conversion
    cv = CountVectorizer()
    cv_train = cv.fit_transform(X_train)
    cv_test = cv.transform(X_test)

    print("\n\nMachine Learning Sentiment Analysis")
    print("\nPrediction with Logistic Regression")
    clothes_preds = lr_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, clothes_preds, "Logistic Regression")

    print("\nPrediction with Logistic Regression after Sampling\n")
    clothes_preds, new_y_test = apply_sampling_for_lr(X_ros, y_ros)
    evaluate_preds(new_y_test, clothes_preds, "Logistic Regression with ROS")

    clothes_preds, new_y_test = apply_sampling_for_lr(X_smote, y_smote)
    evaluate_preds(new_y_test, clothes_preds, "Logistic Regression with SMOTE")

    clothes_preds, new_y_test = apply_sampling_for_lr(X_nm, y_nm)
    evaluate_preds(new_y_test, clothes_preds, "Logistic Regression with Nearest Miss")

    print("\nPrediction with Support Vector Machines")
    clothes_preds = svm_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, clothes_preds, "Support Vector Machines")

    print("\nPrediction with K-Nearest Neighbour")
    clothes_preds = knn_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, clothes_preds, "K-Nearest Neighbour")

    # print("\nPrediction with Multi Perceptron Layer")
    # clothes_preds = mlp_classifier(cv_train, cv_test, y_train)
    # evaluate_preds(y_test, clothes_preds, "Multi Perceptron Layer")

    print("\n\nDeep Learning Sentiment Analysis\n")
    print("\nPrediction with BERT BMUS")
    counter[0] = 0
    sample_number[0] = len(clothes_text)
    print_progress(0, sample_number[0], length=50, printEnd='\r')

    clothes_preds = apply_bert_bmus(clothes_text)
    evaluate_preds(clothes_labels, clothes_preds, "BERT BMUS")

    return


def run_for_food_dataset(apply_bert_bmus):
    print("\n\n\t\t\tAmazon Fine Food Reviews Star Rating Analysis:")
    print("\nLoaded preprocessed dataset:\n")


    # Reading Data
    dishes = pd.read_csv('datasets/preprocessed_dishes.csv', index_col=[0])
    # dishes.drop(dishes.tail(21000).index, inplace=True)
    dishes_text = dishes['Text'].to_numpy()
    dishes_labels = dishes['Score'].to_numpy()


    # Data Visualization
    fig, axs = plt.subplots(ncols=4)
    display(dishes)
    print(dishes.info)

    X_ros, y_ros = ros_sampling(dishes_text, dishes_labels)
    X_smote, y_smote = smote_sampling(dishes_text, dishes_labels)
    X_nm, y_nm = near_miss_sampling(dishes_text, dishes_labels)

    sns.countplot(x=dishes['Score'], ax=axs[0]).set(title='Original Data')
    sns.countplot(x=y_ros, ax=axs[1]).set(title='After ROS')
    sns.countplot(x=y_smote, ax=axs[2]).set(title='After SMOTE')
    sns.countplot(x=y_nm, ax=axs[3]).set(title='After NMS')
    plt.show()


    # Star Rating (Sentiment) Analysis
    X_train, X_test, y_train, y_test = train_test_split(dishes_text, dishes_labels, test_size=0.2)

    # Text to numerical conversion
    cv = CountVectorizer()
    cv_train = cv.fit_transform(X_train)
    cv_test = cv.transform(X_test)

    print("\n\nMachine Learning Sentiment Analysis")
    print("\nPrediction with Logistic Regression")
    dishes_preds = lr_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, dishes_preds, "Logistic Regression")

    print("\nPrediction with Logistic Regression after Sampling\n")
    dishes_preds, new_y_test = apply_sampling_for_lr(X_ros, y_ros)
    evaluate_preds(new_y_test, dishes_preds, "Logistic Regression with ROS")

    dishes_preds, new_y_test = apply_sampling_for_lr(X_smote, y_smote)
    evaluate_preds(new_y_test, dishes_preds, "Logistic Regression with SMOTE")

    dishes_preds, new_y_test = apply_sampling_for_lr(X_nm, y_nm)
    evaluate_preds(new_y_test, dishes_preds, "Logistic Regression with Near Miss")

    print("\nPrediction with Support Vector Machines")
    dishes_preds = svm_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, dishes_preds, "Support Vector Machines")

    print("\nPrediction with K-Nearest Neighbour")
    dishes_preds = knn_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, dishes_preds, "K-Nearest Neighbour")

    # print("\nPrediction with Multi Perceptron Layer")
    # dishes_preds = mlp_classifier(cv_train, cv_test, y_train)
    # evaluate_preds(y_test, dishes_preds, "Multi Perceptron Layer")

    print("\n\nDeep Learning Sentiment Analysis")
    print("\nPrediction with BERT BMUS\n")
    counter[0] = 0
    sample_number[0] = len(dishes_text)
    print_progress(0, sample_number[0], length=50, printEnd='\r')

    dishes_preds = apply_bert_bmus(dishes_text)
    evaluate_preds(dishes_labels, dishes_preds, "BERT BMUS")

    return


def run_for_hotel_dataset(apply_bert_bmus):
    print("\n\n\t\t\tTrip Advisor Hotel Reviews:")
    print("\nLoaded preprocessed dataset:\n")


    # Reading Data
    hotels = pd.read_csv('datasets/preprocessed_hotels.csv', index_col=[0])
    hotels_text = hotels['Review'].to_numpy()
    hotels_labels = hotels['Rating'].to_numpy()


    # Data Visualization
    fig, axs = plt.subplots(ncols=4)
    display(hotels)
    print(hotels.info)

    X_ros, y_ros = ros_sampling(hotels_text, hotels_labels)
    X_smote, y_smote = smote_sampling(hotels_text, hotels_labels)
    X_nm, y_nm = near_miss_sampling(hotels_text, hotels_labels)

    sns.countplot(x=hotels['Rating'], ax=axs[0]).set(title='Original Data')
    sns.countplot(x=y_ros, ax=axs[1]).set(title='After ROS')
    sns.countplot(x=y_smote, ax=axs[2]).set(title='After SMOTE')
    sns.countplot(x=y_nm, ax=axs[3]).set(title='After NMS')
    plt.show()


    # Star Rating (Sentiment) Analysis
    X_train, X_test, y_train, y_test = train_test_split(hotels_text, hotels_labels, test_size=0.2)

    # Text to numerical conversion
    cv = CountVectorizer()
    cv_train = cv.fit_transform(X_train)
    cv_test = cv.transform(X_test)

    print("\n\nMachine Learning Sentiment Analysis")
    print("\nPrediction with Logistic Regression")
    hotels_preds = lr_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, hotels_preds, "Logistic Regression")

    print("\nPrediction with Logistic Regression after Sampling\n")
    hotels_preds, new_y_test = apply_sampling_for_lr(X_ros, y_ros)
    evaluate_preds(new_y_test, hotels_preds, "Logistic Regression with ROS")

    hotels_preds, new_y_test = apply_sampling_for_lr(X_smote, y_smote)
    evaluate_preds(new_y_test, hotels_preds, "Logistic Regression with SMOTE")

    hotels_preds, new_y_test = apply_sampling_for_lr(X_nm, y_nm)
    evaluate_preds(new_y_test, hotels_preds, "Logistic Regression with Near Miss")

    print("\nPrediction with Support Vector Machines")
    hotels_preds = svm_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, hotels_preds, "Support Vector Machines")

    print("\nPrediction with K-Nearest Neighbour")
    hotels_preds = knn_classifier(cv_train, cv_test, y_train)
    evaluate_preds(y_test, hotels_preds, "K-Nearest Neighbour")

    # print("\nPrediction with Multi Perceptron Layer")
    # hotels_preds = mlp_classifier(cv_train, cv_test, y_train)
    # evaluate_preds(y_test, hotels_preds, "Multi Perceptron Layer")

    print("\n\nDeep Learning Sentiment Analysis")
    print("\nPrediction with BERT BMUS")
    counter[0] = 0
    sample_number[0] = len(hotels_text)
    print_progress(0, sample_number[0], length=50, printEnd='\r')

    hotels_preds = apply_bert_bmus(hotels_text)
    evaluate_preds(hotels_labels, hotels_preds, "BERT BMUS")

    return


def ros_sampling(X, y):
    ros = RandomOverSampler(random_state=42)

    cv = CountVectorizer()
    cv_train = cv.fit_transform(X)

    X_ros, y_ros = ros.fit_resample(cv_train, y)

    print('\nRandom Over Sampling\n')
    print('Original dataset shape', Counter(y))
    print('Resample dataset shape', Counter(y_ros), '\n')

    return X_ros, y_ros


def smote_sampling(X, y):
    smote = SMOTE()

    cv = CountVectorizer()
    cv_train = cv.fit_transform(X)

    X_smote, y_smote = smote.fit_resample(cv_train, y)

    print('Synthetic Minority Oversampling Technique\n')
    print('Original dataset shape', Counter(y))
    print('Resample dataset shape', Counter(y_smote), '\n')

    return X_smote, y_smote


def near_miss_sampling(X, y):
    nm = NearMiss()

    cv = CountVectorizer()
    cv_train = cv.fit_transform(X)

    X_nm, y_nm = nm.fit_resample(cv_train, y)

    print('Near Miss')
    print('Original dataset shape:', Counter(y))
    print('Resample dataset shape:', Counter(y_nm), '\n')

    return X_nm, y_nm


def apply_sampling_for_lr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_preds = lr_classifier(X_train, X_test, y_train)

    return y_preds, y_test


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent} % {suffix}', end=printEnd)
    print()

    return


def main():

    # urls = ["https://www.yelp.com/biz/old-ottoman-cafe-ve-restaurant-istanbul",
    #         "https://www.yelp.com/biz/roof-mezze-360-istanbul-2",
    #         "https://www.yelp.com/biz/akdeniz-hatay-sofras%C4%B1-istanbul",
    #         "https://www.yelp.com/biz/ayasofya-m%C3%BCzesi-istanbul",
    #         "https://www.yelp.com/biz/kariye-m%C3%BCzesi-istanbul-2",
    #         "https://www.yelp.com/biz/pera-m%C3%BCzesi-beyo%C4%9Flu?osq=museums",
    #         "https://www.yelp.com/biz/hairvana-istanbul?osq=Hair+Salons",
    #         "https://www.yelp.com/biz/galata-no5-kuaf%C3%B6r-istanbul?osq=Hair+Salons",
    #         "https://www.yelp.com/biz/emre-demircan-hair-studio-istanbul?osq=Hair+Salons",
    #         "https://www.yelp.com/biz/m%C4%B1s%C4%B1r-%C3%A7ar%C5%9F%C4%B1s%C4%B1-istanbul-4",
    #         "https://www.yelp.com/biz/kapal%C4%B1%C3%A7ar%C5%9F%C4%B1-istanbul-2",
    #         "https://www.yelp.com/biz/zorlu-avm-istanbul-2"]
    #
    # for url in urls:
    #     all_reviews = scrap_data(url)
    #
    #     counter = 0
    #     for index in range(len(all_reviews)):
    #         print(all_reviews[index])
    #         counter += 1
    #
    #     print(counter)

    # index = 0
    # control = True
    # input = "& start = "
    # while control:
    #
    #
    #     index += 10

    # df = pd.DataFrame(np.array(all_reviews), columns=['review'])
    #
    # for index, row in df.iterrows():
    #     print(row)

    # prepare_original_data()

    # Sample Text Star Rating Prediciton
    # sample_text = "It was perfect but could have been better, wonderful"
    # example_bmus(sample_text)


    # Applying Star Rating Prediction and Evaluation

    bert_bmus_v = np.vectorize(bert_bmus)

    # run_for_clothing_dataset(bert_bmus_v)

    run_for_food_dataset(bert_bmus_v)

    # run_for_hotel_dataset(bert_bmus_v)

    # example_bmus("")


    return


if __name__ == '__main__':
    start_time = time.monotonic()

    # warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print("\nAvailable hardware to compute:", device, '\n')

    main()

    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)

    print("\n\nExecution time:", total_time)

    winsound.Beep(440, 2000)
