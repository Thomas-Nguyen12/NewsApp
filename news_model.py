#!/usr/bin/env python3
import time
print (time.asctime())
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
import numpy as np
import sklearn
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# This module refers to k-nearest neighbors for multi-label
from catboost import CatBoostClassifier
import xgboost
## the ARAAM model is a neural network for large scale text classification 
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix 
import re 
import joblib 
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
""" 
This program will build a model for my news data to predict
trends over time 
"""





def load_data(): 
    df = pd.read_csv("/Users/thomasnguyen/news_project/scraped_news_2004_2025.csv")
    # 1. checking for 
    """
    CLEANING:
    1. missing values
    2. erroeneous data
        - This can include rows that have [removed] 
    3. column names
    
    I can use the scikit-learn column transformer and pipeline modules
    
    
    ENCODING
    - for this, I should have already used PCA in my exploratory data anlaysis 
    
    
    """
    return df 


def clean_data(data): 
    
    # removing the missing row (by date) 
    data.index = pd.to_datetime(data.date) 
    
    
    # dropping the unecessary column 
    
    # removing the "edit, history, watch" 
    
    
    
    # removing the row with the missing data (June 2, 2025). I first need to identify this row. Cleaned format is year, month, day
    # I need to
    # i need to detect the rows that have missing topics or 
    
    
    row_to_remove = data[data.text == "edit,history,watch"].index
    data = data.drop(row_to_remove, axis=0)
    
    data.drop(['date'], axis=1, inplace=True)
    # 
    
    data.text = data.text.str.replace("^edit,history,watch,", "", regex=True) 
    # I need to make sure that there are no topics that are the same but spelled 
    # slightly differently 
    
    ## topics = ['Armed conflicts and attacks', 'Arts and culture',
    #   'Business and economy', 'Disasters and accidents',
    #   'Health and environment', 'International relations',
    #   'Law and crime', 'Politics and elections',
    #   'Science and technology', 'Sports', 'Businesses and economy',
    #   "Politics and elections'", 'Science',
    #   'Armed attacks and conflicts', 'Other', 'Royalty',
    #   'Science and Technology', 'Politics and economics',
    #   'Disasters and Accidents', 'Law and Crime',
    #   'Science and technology ', 'Entertainment', 'Arts and culture ',
    #   'Attacks and armed conflicts', 'Arts and Culture ',
    #   ' Disasters and accidents ']
    
    
    # removing whitespaces
    
    """ 
    array(['Armed conflicts and attacks', 'Arts and culture',
       'Disasters and accidents', 'Health and environment',
       'International relations', 'Law and crime',
       'Politics and elections', 'Sports',
       'Science and technology and technologyand technology',
       'Business and economy', 'Science and technology and technology',
       'Other', 'Politics and economics', 'Royalty',
       'Science and technology and technologyand Technology',
       'Science and technology and technologyand technology ',
       'Entertainment', 'Attacks and armed conflicts'], dtype=object)
    
    """
    
    
    """ 
    array(['Armed conflicts and attacks', 'Arts and culture',
       'Disasters and accidents', 'Health and environment',
       'International relations', 'Law and crime',
       'Politics and elections', 'Sports',
       'Science and technology and technologyand technology',
       'Business and economy', 'Science and technology and technology',
       'Other', 'Politics and economics', 'Royalty',
       'Science and technology and technologyand Technology',
       'Science and technology and technologyand technology ',
       'Entertainment'], dtype=object)
    """
    
    """ 
    
    
 
 
 'Science and technology and technologyand technology'
 'Science and technology and technology'
  
 'Science and technology and technologyand Technology'
 'Science and technology and technologyand technology ']
    """
    
    
    # I can use a lambda function 
    
    

    data.topic = data.topic.str.replace("Science and technology and technologyand technology", "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technology", "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technologyand Technology", "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technologyand technology ", "Science and technology", regex=True)
    
    data.topic = data.topic.str.replace("Attacks and armed conflicts", "Armed conflicts and attacks", regex=True) 
    data.topic = data.topic.str.replace('Science and technology and technology', "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technologyand technology'", "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technologyand Technology", "Science and technology", regex=True) 
    data.topic = data.topic.str.replace("Science and technology and technologyand technology", "Science and technology", regex=True) 
    
    
    data.topic = data.topic.str.replace("Armed attacks and conflicts", "Armed conflicts and attacks", regex=True) 
    
    data.topic = data.topic.str.replace("Arts and culture ", "Arts and culture", regex=True) 
    data.topic = data.topic.str.replace("Arts and Culture ", "Arts and culture", regex=True)
    
    data.topic = data.topic.str.replace("Businesses and economy", "Business and economy", regex=True) 
    
    data.topic = data.topic.str.replace(" Disasters and accidents ", "Disasters and accidents", regex=True)
    data.topic = data.topic.str.replace("Disasters and Accidents", "Disasters and accidents", regex=True)
    
    data.topic = data.topic.str.replace("Law and Crime", "Law and crime", regex=True) 
    
    data.topic = data.topic.str.replace("Politics and elections'", "Politics and elections", regex=True) 
    
    data.topic = data.topic.str.replace("Science", "Science and technology", regex=True) 
    
    data.topic = data.topic.str.replace("Science and Technology", "Science and technology", regex=True) 
    
    data.topic = data.topic.str.replace("Science and technology ", "Science and technology", regex=True) 
    
    
    data.topic = data.topic.str.replace("Science", "Science and technology", regex=True)
    data.topic = data.topic.str.replace("Disasters and incidents", "Disasters and accidents", regex=True)
    
    # replacing the weirdly labelled column labels 
    
    return data 
    
    
    
def extract_topics(new_df): 
    # I can extract the topic for each row and keep the date
    # To do this, I should keep track of how many topics there are and match the number of years
    # i can use tuples to do this 
    
    new_df.topic = new_df.topic.str.replace("Science and technology and technologyand technology", "Science and technology", regex=True) 
    new_df.topic = new_df.topic.str.replace("Science and technology and technology", "Science and technology", regex=True) 
    new_df.topic = new_df.topic.str.replace("Science and technology and technologyand Technology", "Science and technology", regex=True) 
    new_df.topic = new_df.topic.str.replace("Science and technology and technologyand technology ", "Science and technology", regex=True)
    
    """ 
    array(['Armed conflicts and attacks', 'Arts and culture',
       'Disasters and accidents', 'Health and environment',
       'International relations', 'Law and crime',
       'Politics and elections', 'Sports', 'Science and technology',
       'Business and economy', 'Other', 'Politics and economics',
       'Royalty', 'Science and technologyand Technology',
       'Science and technology ', 'Entertainment'], dtype=object)
    """
    new_df.topic = new_df.topic.str.replace("Science and technology ", "Science and technology", regex=True) 
    new_df.topic = new_df.topic.str.replace("Science and technologyand Technology", "Science and technology", regex=True) 
    
    
    
    new_df['topic_cleaned'] = new_df.topic.str.split(",") 
    
    
    # ----- CREATING A NEW DATAFRAME HERE

    new_df2 = new_df.drop(['text'], axis=1)
    new_df2['date'] = new_df2.index
    exploded_df = new_df2.explode('topic_cleaned')

    # Select only the date and the individual topic
    result_df = exploded_df[['date', 'topic_cleaned']].rename(columns={'topic_cleaned': 'topic'})
    print ("Result df")
    print(result_df)
    result_df = pd.DataFrame(result_df)
    result_df_grouped = result_df.groupby([result_df.index.year, result_df.index.month, result_df.topic])['topic'].count()

    result_df_grouped_df = pd.DataFrame(result_df_grouped)
    result_df_grouped_df.rename({"topic": "count"}, axis=1, inplace=True)
    # Step 1: Rename duplicate index levels (if necessary)
    result_df_grouped_df.index.set_names(['year', 'month', 'topic'], inplace=True)

    # Step 2: Reset the index
    flat_df = result_df_grouped_df.reset_index()

    # Now you have columns: 'year', 'month', 'topic', 'count'
    print(flat_df.head())
    flat_df['sep'] = '-' 

    flat_df['year_month'] = pd.concat([flat_df.year, flat_df['sep'], flat_df.month], ignore_index=True)
    flat_df['year_month'] = flat_df['year'].astype(str) + '-' + flat_df['month'].astype(str).str.zfill(2)
    flat_df.drop(['sep'], axis=1, inplace=True)
    
    return flat_df

def extract_text(data): 
    pass 

    

    


def preprocess_data(data): 
    
    """ 
    The objective is to track news data over time. 
    
    To do this, I will model the rate of news reports for each topic.
    
    To preprocess, I will need to convert the "topic" column into a one suitable
    for multilabel classification 
    
    The multilearn library provides a way to complete this task. I will
    need a list of the unique topics 
    
    I will need to search of the "unqiue" topic in each row 
    
    There should be 14 unique topics. This could be a very computationally 
    Expensive task. 
    
    
    
    """
    
    # putting all the words on lower case 
    print ("Lowering the text column")
    data['text'] = data['text'].str.lower() 
    
    
    topics = ['Armed conflicts and attacks', 'Arts and culture',
       'Business and economy', 'Disasters and accidents',
       'Health and environment', 'International relations',
       'Law and crime', 'Politics and elections',
       'Science and technology', 'Sports', 'Other', 'Royalty',
       'Politics and economics', 'Entertainment']
    
    print ("adding all of the topics as separate columns...")
    data[[*topics]] = None
    
    
    print ("This is the dataframe...")
    ## creating the new columns 
    print (data.head()) 
    
    print ("These are the columns...") 
    print (data.columns)
    
    
    print ("These are the dimensions...") 
    print (data.shape) 
    
    print ("Searching for if the topic is in the topic column... If yes, a 1 is added to the corresponding topic. If no, a 0")
    #for index in range(len(data.topic)): 
    #    for category in topics: 
    #        topic_value = str(data.topic.iloc[index])
    #        
    #        # I could also do a regex search 
    #        if re.search(category, data.topic.iloc[index]):
    #            
    #            data[category].iloc[index] = 1
    #        else: 
    #            
    #            data[category].iloc[index] = 0 
                
                
    for category in topics:
        data[category] = data['topic'].astype(str).apply(
            lambda x: 1 if re.search(category, x, re.IGNORECASE) else 0
        )
    
    
    # preprocessing the text (using vectorizer)
    
    
    
    ## encoding the topic column 
    
    # Do i need to encode the target variable? 
    
    
    
    
    
    # creating the new columns 
    return data 
    
    





def build_model(data): 
    """
    This will be a multi-label classification model 
    to classify news reports.
    
    For this, I will use binary relevance as it is a very simple technique
    and very interpretable.
    
    """
    
    # I need to save the vectorizer
    vectoriser = TfidfVectorizer(stop_words=list(stop_words), lowercase=True)
    
    # vectorising the text 
    topics = ['Armed conflicts and attacks', 'Arts and culture',
       'Business and economy', 'Disasters and accidents',
       'Health and environment', 'International relations',
       'Law and crime', 'Politics and elections',
       'Science and technology', 'Sports', 'Other', 'Royalty',
       'Politics and economics', 'Entertainment']
    
    
    # I need to make sure I drop all of the columns 
    print ("Forming the X and Y...")
    X = data.text
    print ("#--------------------")
    
    print ("#--------------------")
    
    
    print (f"Shape of X: {np.shape(X)}")
    
    print ("Creating the target variable...")
    y = data[[*topics]].values
    y = csr_matrix(y, dtype=np.int64)
    
    # ------ converting y into a matrix 
    
    
    
    # ------ the vectorizer should be fit on the training dataset first 
    
    
    
    
    print ("Splitting the dataset...")
    
    # ------ Make sure there is a consistent number of samples 
    # ------ The vectoriser should be fit within the clean dataframe
    
    # ----- Found input variables with inconsistent numbers of samples: [1001, 14014]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    
    # ------ Vectorising the training and test dataset 
    joblib.dump(Y_test, "Y_test_original.pkl")
    
    print ("Vectorising...")
    # ------ should only vectorise the training as it prevents data leakge
    X_train_tfidf = vectoriser.fit_transform(X_train) 
    X_test_tfidf = vectoriser.transform(X_test) 
    
    joblib.dump(vectoriser, 'tfidf_vectoriser.pkl')
    
    
    
    
    
    
    
    print ("Creating the Classifier...")
    # ------ scipy.sparse does not support dtype object 
    # creating an optimised model
    classifier = xgboost.XGBClassifier(
    )
    print ("Optimising...")

    classifier = BinaryRelevance(classifier)
    grid_search = classifier
    
    
    # train
    print ("Fitting the Classifier...")
    
    # ------ Make sure that datatypes of the training and testing are right
    # ------ Check datatypes of X_train and Y_train 
    # ------ make sure the y_train is in int format 
    
    # ------ why is the X train shape (1,1)?
    
    
    print (f"Shape of X_train_tfidf: {np.shape(X_train_tfidf)}")
    print (X_train_tfidf)
    print ("-------------------------")
    
    print (f"Shape of Y_train: {np.shape(Y_train)}")
    print ("Fitting the optimiser...")
    grid_search.fit(X_train_tfidf, Y_train)
    
    
    print ("Finished Optimising...")
    #    predict
    
    final_model = grid_search
    print ("creating predictions...") 
    
    predictions = final_model.predict(X_test_tfidf) 
    
    print (f"Shape of the predictions: {np.shape(predictions)}")
    print (f"predictions...: {predictions}")
    print (f"Accuracy: {accuracy_score(predictions, Y_test)}")
    print (f"Weighted f1 score: {f1_score(predictions, Y_test, average="weighted")}")
    print (f"Weighted Precision: {precision_score(predictions, Y_test, average='weighted')}")
    print (f"Weighted Recall: {recall_score(predictions, Y_test, average='weighted')}")
    
    
    
    
    
    # ----- PLOTTING THE CONFUSION MATRIX 
    
    
    
    
    
    return final_model, predictions, Y_test, X_test_tfidf
    
    ## Metrics 
    
def plot_roc_curve(predictions, Y_test): 
    
    # ------ Need to use the Y true and predictions
    
    pass 

    
def plot_confusion_matrix(model, predictions, Y_test): 
    cm = confusion_matrix(Y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
    disp.plot() 
    plt.show() 







# running all of the scripts

def main(): 
    try: 
        
        print ("Loading the data...")
        data = load_data() 
        print ("Cleaning the data...")
        clean_df = clean_data(data) 
        clean_df.to_csv("clean_df.csv") 
        print ("Extracting the topics...") 
        extracted_topics = extract_topics(clean_df) 
        extracted_topics.to_csv("extracted_topics.csv")
        print ("Preprocessing the data...") 
        preprocessed_data = preprocess_data(clean_df) 
        print ("Building the model...")
        model, predictions, Y_test, X_test_tfidf = build_model(preprocessed_data) 
        print ("Plotting metrics...") 
        
        #plot_confusion_matrix(model, predictions, Y_test)
        # ---- Saving the model 
        print ("Saving the model...") 
        joblib.dump(model, "news_model.pkl")
        print ("Saving the dataset...") 
        joblib.dump(preprocessed_data, 'preprocessed_data.pkl')
        
        ## saving the output of Y_test and predictions
        joblib.dump(predictions, 'predictions.pkl')
        joblib.dump(Y_test, 'Y_test.pkl')
       
        joblib.dump(X_test_tfidf, 'X_test_tfidf.pkl')
        
        
    except Exception as e: 
        print ("There was an error...") 
        print (e)
    else:
        print ("Model successfully built...") 
    finally:
        print ("Exiting the program...") 
        
        
    
    
    

if __name__ == "__main__": 
    main() 
    print (time.asctime())