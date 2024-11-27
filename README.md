# Reddit-Stocks-Insights
This project focuses on analyzing sentiment from stock-related discussions on Reddit to understand public perception and trends in the financial market. We trained a sentiment analysis model using data scraped from Reddit's API, targeting posts from subreddits such as r/stocks and others.

The goal of this project is to extract insights about market sentiment by analyzing post titles and scores (upvotes minus downvotes). The sentiment analysis model was developed to classify posts into positive, negative, or neutral categories, providing valuable information for investors and market analysts.

The project includes the following steps:

1. Scraping stock-related data from Reddit using its official API.<br>

2. Preprocessing the scraped data .<br>

3. Training and evaluating a  model.<br>

4. Generating insights and visualizations from the model's predictions.
-----
### Data Scraping Process
-----
The data for this project was scraped from Reddit using the Reddit API. Below are the steps followed to collect the data:

**1. Creating a Reddit Developer Account**
+ A Reddit account was created, and the developer portal was accessed to create an application.
+ API credentials, including the client_id, client_secret, and user_agent, were obtained for making authenticated API calls.

**2. Using the Reddit API**
+ The PRAW Python library was used to interact with the Reddit API.
+ Subreddits related to stock discussions were queried to collect relevant posts.
+ The following fields were extracted:
    * Title: The main text of the Reddit post.
    * Author: The username of the post creator.
    * Score: The upvotes minus downvotes.
    * URL: The direct link to the post.
    * Comments: The number of comments on the post.
    * Created_at: The timestamp of post creation.
      
**3. Storing the Raw Data**
+ The scraped data was stored in a file named stock_data.csv for further processing.
  
For the detailed code implementation, refer to the respective script in the repository.

--------
### Data Preprocessing
----------
After scraping the data from Reddit, the following preprocessing steps were performed to prepare it for sentiment analysis:
**1. Checking for Null Values**
+  The dataset was checked for missing or null values in important columns (e.g., title, score).<br>
 + Rows with null values were removed to maintain the integrity and consistency of the dataset.<br>
 
**2. Stemming Post Titles**<br>
 + The title column was processed using stemming to reduce words to their root form.<br>
 + This normalization step helps in improving the accuracy of the sentiment analysis model.<br>
 
**3.  Mapping Sentiment Labels**<br>
+ Sentiment labels were assigned to each post based on its score, using the following mapping:<br>
 + Positive: Posts with a score of 15 or higher.<br>
+  Neutral: Posts with a score between 5 and 14.<br>
 + Negative: Posts with a score below 5.<br>
+ A new column, sentiment_label, was created to store these sentiment labels.<br>

**4. Final Dataset**<br>
+ The final dataset after preprocessing includes the following columns:<br>
+ title: The original title of the Reddit post.<br>
+ author: The author of the post.<br>
+ score: The difference between upvotes and downvotes for the post.<br>
+ upvotes: The total number of upvotes on the post.<br>
+ url: The direct URL of the Reddit post.<br>
+ created_at: The timestamp when the post was created.<br>
+ comments: The total number of comments on the post.<br>
+ sentiment_label: The sentiment classification (Positive, Neutral, or Negative).<br>
+ stemmed_content: The stemmed version of the post's title.<br>

**5. Processed Dataset**<br>
+ The cleaned and preprocessed data was saved into the file stock_data1.csv for further analysis and model training.<br>
+ Readers can refer to the included code for the full implementation of the preprocessing steps.<br>
----------
### Model Training
----------
After preprocessing the data, the next step involved training a sentiment analysis model. The process included feature extraction, model training, and model evaluation:

 **1. Feature Extraction**<br>
+ Text data from the stemmed_content column was used for feature extraction.<br>
 + The TF-IDF (Term Frequency-Inverse Document Frequency) method was applied to convert the text into numerical features, 
 capturing the importance of words in the dataset.<br>
 + The sentiment labels were encoded into numerical values. The labels were mapped as follows:<br>
 * Negative: 0<br>
 * Neutral: 1<br>
 * Positive: 2<br>
 + The encoded labels were stored in a new column sentiment_label_encoded.<br>

**2. Model Training**<br>
+  Logistic Regression was used to train the sentiment analysis model.<br>
 + The dataset was split into training and testing sets, where the training set was used to build the model, and the testing 
   set was used to evaluate its performance.<br>
 + The model was trained to predict sentiment labels (Positive, Neutral, or Negative) based on the extracted features and the 
 encoded labels.<br>

**3. Model Evaluation**<br>
 + The model’s performance was evaluated using key metrics such as accuracy, precision, recall, and F1-score.<br>
 + These metrics helped assess the model's effectiveness in predicting sentiment accurately.<br>
+ Once the model was trained and evaluated, it was ready for making predictions on new data.<br>

For more details on the model implementation and evaluation, please refer to the code files in the repository.<br>

------------
### True vs. Predicted Labels Visualization
------------
To evaluate the performance of the model visually, we created a scatter plot comparing the true labels with the predicted labels for a subset of the test data. The following steps were performed:

+ Selecting a Subset: The first 20 samples from the test set are used for visualization.
+ Scatter Plot: A scatter plot is generated where:
+ True Labels are shown in blue.
+ Predicted Labels are shown in orange (marked with an 'x').<br>

![Screenshot (123)](https://github.com/user-attachments/assets/0d0c66ae-48ad-4dd5-95ef-3f0c7224644e)




This visualization allows for a quick comparison between the model’s predictions and the actual sentiment labels, helping to assess the model’s performance on a small subset of the test data.

-----------
## Using the Trained Model
-----------
The trained Logistic Regression model is saved in the file trained_model.sav. Follow these steps to use the model:

**1. Clone the Repository**
+ Clone this repository to your local machine to access the necessary files.

**2. Install Dependencies**
+ Ensure that you have all the required Python libraries installed. You can refer to the code to identify the libraries used (e.g., sklearn, pickle, etc.).

**3. Load the Model**
+ The trained model is saved in the file trained_model.sav. Load this model into your Python environment to make predictions.

**4. Prepare Input Data**
 To use the model, you need to process the input text data in the same way as during training:
+ Create and Fit the TF-IDF Vectorizer
+ Use a TF-IDF vectorizer, configured with parameters similar to those used during training. The vectorizer needs to be fitted on your training data (or a similar dataset) before transforming any new input data.
+ Transform the Input Data
+ Apply the fitted TF-IDF vectorizer to the input text to convert it into numerical form for model prediction.

**5. Make Predictions**
+ Once the model is loaded and the input text is processed using the TF-IDF vectorizer, use the trained model to predict the sentiment label of the text.
+ Sentiment Mapping
+ The model will output the following encoded labels:
0 for Negative
1 for Neutral
2 for Positive
+ Map these labels back to their corresponding sentiment categories (Negative, Neutral, Positive).

**6. Notes**
+ Refer to the provided Python scripts for detailed instructions on text preprocessing, TF-IDF vectorization, and how to load the model.
+ Ensure that you preprocess the input data in the same manner as the training data for accurate predictions.
-----
### Technologies Used
-----
**1. Reddit API**: Used to scrape stock-related posts and data from Reddit, extracting valuable information for sentiment analysis.

**2. Python**: The core programming language used throughout the project for data scraping, preprocessing, model training, and evaluation.

**3. Requests & PRAW**: The Requests library is used for making HTTP requests to the Reddit API, while PRAW (Python Reddit API Wrapper) simplifies the process of interacting with Reddit data.

**4. Pandas**: A powerful library for data manipulation and analysis, used to store and clean the scraped data.

**5. NLTK & Spacy**: Libraries used for natural language processing tasks, such as stemming and tokenization, to prepare the text data.

**6. Scikit-learn**: A machine learning library used for feature extraction (TF-IDF), training the Logistic Regression model, and evaluating its performance.

**7. Pickle**: Used to serialize the trained model (Logistic Regression) and save it to disk for later use.

**8. Matplotlib & Seaborn**: Used for visualizations, such as plotting evaluation metrics and model performance.

**9. Jupyter Notebook**:  A tool for data exploration, experimentation, and visualization during the development phase.
