# Analysing Reddit Comments

> This project was intended to analyze Reddit comments to discover trends throughout the Reddit community. Is there a relationship between subreddits and score? Does having a high Reddit score mean your comment will also have a high gilded score? Analyses were completed with a weeks’ worth of Reddit data which determined that there is a correlation between score and text and there is a correlation between score and subreddit.

## Introduction
Reddit is an online community where users can post content into specific sub-communities called ‘subreddits’, also thought of as a topic, that other user can view within the subreddit. Users can ‘upvote’, add one point, or ‘downvote’, subtract one point, posts which will change the score of that post. Those posts with a high positive score are more easily seen throughout the community unlike those with a low score.

Users can also comment on and discuss posted content in each post’s ‘thread’ and comments can also be upvoted or downvoted as well as ‘gilded’ to show the community’s reaction to them. Reddit also uses other metrics to score comments such as ‘controversiality’ and ‘distinguished’.
A dataset containing all comments made between Feb. 18, 2019, to Feb. 24, 2019, was used for the project. This Reddit dataset includes detailing for each comment; the subreddit it belongs to, the user who posted it, the number of upvotes and downvotes it received as well as other information as will be discussed later in this report.


## Team KWMZ

[Jon Kang](https://github.com/skang77e)

[Siqi(Eva) Wang](https://github.com/evaevaevawang)

[Jingyuan Meng](https://github.com/Mengjingyuan)

[Gabriella Zakrocki](https://github.com/zakrocki-gabriella)


## Code files
Data Exploration: `data_exploration.ipynb`

Predict the Score Using Comment Text: `Predicting Score Using Comment Text.ipynb`

Find Top score + Predict Number of Gilded based on Score: `GildedPrediction-fromScore.ipynb`

Guilded counts + Predict Subreddit based on Comment Text: `Gilded_counts-Subreddit_ranks-ML_Predict_Subreddit(classification).ipynb`

## Methods
- How you cleaned, prepared the dataset with samples of intermediate data?
  1. Imported the dataset into AWS S3, read as JSON dataframe.
  ```sh 
  reddit_df = spark.read.json(s3_URL)
  ```
  2. Defined and print the schema.
  
  ```sh 
  reddit_df.printSchema()
  ```
  
  3. Dropped the columns that are not related to project purpose.
  ```sh
    columns_to_drop = ['author_cakeday', 
                   'author_created_utc',
                   'author_flair_background_color',
                   'author_flair_richtext',
                   'author_flair_template_id',
                   'author_flair_text_color',
                   'author_flair_type',
                   'author_fullname',
                   'author_patreon_flair',
                   'can_gild',
                   'can_mod_post',
                   'collapsed',
                   'collapsed_reason',
                   'gildings',
                   'is_submitter',
                   'no_follow',
                   'permalink',
                   'quarantined',
                   'removal_reason',
                   'send_replies',
                   'stickied',
                   'subreddit_name_prefixed',
                   'subreddit_type'
                  ]
  reddit_df = reddit_df.drop(*columns_to_drop)
  ```
  4. Print out samples from the data.
  ```sh
  reddit_df.show(3)
  ```
  5. Cache the dataset to save time.
  ```sh
  reddit_df.cache()
  ```
- Tools you used for analyzing the dataset and the justification (tools, models, etc.)?
  - Predict the Score Using Comment Text (Regression)
    - Tokenizer to convert the comment bodies to arrays
    - Remove stop words from words column
    - CountVectorizer
    - HashingTF 
    - RandomForest Model
  - Predict Number of Gilded based on Score (Regression)
    - VectorAssembler
    - MLib Pipeline
    - randomSplit
    - RandomForest Model
  - Predict Subreddit based on Comment Text (Classification)
    - StringIndexer
    - RegexTokenizer
    - StopWordsRemover
    - HashingTF
    - IDF
    - MLib Pipeline
    - RandomForest Model

- How did you model the dataset, what techniques did you use and why?

  We made three machine learning tasks: 
  
  Regression:
    - Predict the score based on the body of comment 
    - Predict gilded score based on subreddit score
    
  Classification:
    - Predict the subreddit type based on the content of comment 
  
  Our main goal of these machine learning jobs is to try using machine learning on a large dataset, since there will be a huge amount of computation, but not using the best model and then get the most accurate results. Thus, we choose one model that will be used by anyone who is doing machine learning tasks. Besides, the regression model is not complex, so we don't need to wait forever for the computation.
  
  Here are some essential steps we did during the machine learning process:
    - Tokenize words and remove stop words. 
    - Implement two methods in organizing dataset: 
      - Method1 (CV) is creating a bag of words.
      - Method2 (HTF) is creating a bag of words and placing similar words in the same bag. Then use RandomForestRegressor on a train set and predict score based on a test set.
      - Calculate TF-IDF, which is used to put less emphasis on words that appear more frequently.
      - Simplify the third prediction by just using top five themed subreddit, which means the subreddit has the specific meanings, but not the words created by the users. The model we used is RandomForestClassifier.


- Did you have a hypothesis that you were trying to prove?

  We believed that there would be a positive relationship between the Reddit score and text within a comment. Politics, sports, news, and AskReddit were thought to be the most popular subreddit topics. It was also expected that the Reddit score would have a positive impact on the gilded score of a comment.
  
- Did you just visualize the dataset, and if so, why?

  We made several visualizations of the features that may be helpful for understanding the Reddit platform, such as 'Top 10 Highest/Lowest Scoring Subreddit', 'Gild ranking by subreddit', and 'Gild ranking of the author'. Then, we find out that 'AskReddit' is the most popular topic in Reddit. Based on the popularity of subreddit, from which we identify the top five subreddit for our third prediction. Besides, the visualizations tell us that politics and sports are the most popular topic in the US.
  
## Results/Conclusions

- What did you find and learn?

  For Data Exploration, we have been able to get results for average score, average controversiality and average gilded score from our data. Thinking through more, we have computed the Max/Min score and Max/Min gilded score and also ranked top 10 controversial subreddits and scored subreddits with graphs to show their numbers. To compare the results, the top 10 least scored subreddits are also being listed here with its bar chart. The total number of Reddit gold, positive, and negative comments are also displayed as numbers. The interesting fact we’ve gotten is that the least top 10 comments with the least score, which looks like comments do not convey any interesting information. 

  For Predict the Score Using Comment Text, we have explored the relationship between Score and Text themselves. After tokenizing the words and removing stop words, we can move further in analyzing the text. By creating two bagging methods, we have stored text in Method 1(CV) with creating a bag of words found in the body of comments and Method 2(HTF) with creating a bag of words found in the body of comments but placing similar words in the same bucket. Applying Random Forest Regressor, we are able to predict the score depends on text. The highest score is about 8.445206051243071 and words are like Vaguely, Remember and more. 

  For Find Top score + Predict Number of Gilded based on Score, we have found from our dataset that out of all of comments only 0.0298% of the data has a gilded score more than 0. Then, we applied Random Forest Regressor to make predictions based on reddit score. The machine learning model worked quite well with a very low RMSE. The only issue is that because there are so few gilded comments, we technically can not logically predict the gilded score from the comment score. The comment with the highest score (53571) from our analysis is provided: “My roommate would get up and steal the shower as soon as he heard my alarm go off.  6 weeks later, I had him waking up to shower at 4:30.  I would just turn off my pavlovian alarm and go back to sleep for another 3 hours waiting for the real alarm.” 

  For Gilded counts + Predict Subreddit based on Comment Text, we got the rank about different subreddit types and authors with highest gilded score received. Before doing the machine learning, we firstly extract the top 5 themed subreddit from the data. Since there is a huge amount of types and many of them have no specific meanings, so it's hard to predict them. Besides, we want to simplify the process, so we choose to predict the top five themed subreddit, the number of comments is 2602191. Then we applied Random Forest Classifier to predict subreddit based on comment text with accuracy of 0.558603.

- How did you validate your results?

  To validate our results, we can compare our results with the new amount of data source and make sure they have enough similarities with the data we have collected. For example, trying to have a similar date range, collecting methods, and more. Since our initial analysis focuses on data from Feb. 18, 2019, to Feb. 24, 2019, we may be able to find a similar data source in Feb 2019 and validate our results then. 

## Future work

  As for the improvement, we can use different algorithms in implementing the model and achieve goals with better accuracy. For example, use KNN to find the clusters, which can help us identify the common features. Then, by adjusting k values, the accuracy can improve some. 

## Bibliography/References

## Division of labor
- Jon Kang: Import data to S3, and data_exploration.ipynb
- Siqi(Eva) Wang: Predicting Score Using Comment Text.ipynb
- Jingyuan Meng: Gilded_counts-Subreddit_ranks-ML_Predict_Subreddit(classification).ipynb
- Gabriella Zakrocki: GildedPrediction-fromScore.ipynb
