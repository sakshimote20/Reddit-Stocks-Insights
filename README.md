# Reddit-Stocks-Insights
This project focuses on analyzing sentiment from stock-related discussions on Reddit to understand public perception and trends in the financial market. We trained a sentiment analysis model using data scraped from Reddit's API, targeting posts from subreddits such as r/stocks and others.

The goal of this project is to extract insights about market sentiment by analyzing post titles and scores (upvotes minus downvotes). The sentiment analysis model was developed to classify posts into positive, negative, or neutral categories, providing valuable information for investors and market analysts.

The project includes the following steps:

1. Scraping stock-related data from Reddit using its official API.<br>

2. Preprocessing the scraped data for sentiment analysis.<br>

3. Training and evaluating a sentiment analysis model.<br>

4. Generating insights and visualizations from the model's predictions.

# Data Scraping Process
1. Set Up a Reddit Account<br>
Create an account on reddit.com if you don’t already have one.<br>
Log in to your Reddit account.<br>
2. Create a Reddit App for API Access<br>
Go to Reddit App Preferences.<br>
Click on "Create App" or "Create Another App" at the bottom of the page.<br>
Fill in the required details:<br>
   App type: Select "Script" (intended for personal use).<br>
   Name: Enter a name for your app.<br>
   Redirect URI: Use http://localhost (this is mandatory but not used for scripts).<br>
Once created, note down the Client ID (found under your app's name) and Secret Key (generated for the app).<br>
3. Install Necessary Libraries refer data scraping file for more details.
# Data Preprocessing
After scraping the data from Reddit, the following preprocessing steps were performed to prepare it for sentiment analysis:
1. Checking for Null Values
 The dataset was checked for missing or null values in important columns (e.g., title, score).<br>
 Rows with null values were removed to maintain the integrity and consistency of the dataset.<br>
 
2. Stemming Post Titles<br>
 The title column was processed using stemming to reduce words to their root form.<br>
 This normalization step helps in improving the accuracy of the sentiment analysis model.<br>
 
3.  Mapping Sentiment Labels<br>
 Sentiment labels were assigned to each post based on its score, using the following mapping:<br>
 Positive: Posts with a score of 15 or higher.<br>
 Neutral: Posts with a score between 5 and 14.<br>
 Negative: Posts with a score below 5.<br>
A new column, sentiment_label, was created to store these sentiment labels.<br>

4. Final Dataset<br>
The final dataset after preprocessing includes the following columns:<br>
title: The original title of the Reddit post.<br>
author: The author of the post.<br>
score: The difference between upvotes and downvotes for the post.<br>
upvotes: The total number of upvotes on the post.<br>
url: The direct URL of the Reddit post.<br>
created_at: The timestamp when the post was created.<br>
comments: The total number of comments on the post.<br>
sentiment_label: The sentiment classification (Positive, Neutral, or Negative).<br>
stemmed_content: The stemmed version of the post's title.<br>

5. Processed Dataset<br>
The cleaned and preprocessed data was saved into the file stock_data1.csv for further analysis and model training.<br>
Readers can refer to the included code for the full implementation of the preprocessing steps.<br>

# Model Training
After preprocessing the data, the next step involved training a sentiment analysis model. The process included feature extraction, model training, and model evaluation:

 1. Feature Extraction<br>
 Text data from the stemmed_content column was used for feature extraction.<br>
 The TF-IDF (Term Frequency-Inverse Document Frequency) method was applied to convert the text into numerical features, 
 capturing the importance of words in the dataset.<br>
 The sentiment labels were encoded into numerical values. The labels were mapped as follows:<br>
 Negative: 0<br>
 Neutral: 1<br>
 Positive: 2<br>
 The encoded labels were stored in a new column sentiment_label_encoded.<br>

2. Model Training<br>
 Logistic Regression was used to train the sentiment analysis model.<br>
 The dataset was split into training and testing sets, where the training set was used to build the model, and the testing 
 set was used to evaluate its performance.<br>
 The model was trained to predict sentiment labels (Positive, Neutral, or Negative) based on the extracted features and the 
 encoded labels.<br>

3. Model Evaluation<br>
 The model’s performance was evaluated using key metrics such as accuracy, precision, recall, and F1-score.<br>
 These metrics helped assess the model's effectiveness in predicting sentiment accurately.<br>
 Once the model was trained and evaluated, it was ready for making predictions on new data.<br>

For more details on the model implementation and evaluation, please refer to the code files in the repository.<br>
