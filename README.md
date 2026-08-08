<H3>NAME: Piritharaman R</H3>
<H3>REGISTER NO: 212223230148</H3>
<H3>DATE: 08/08/26</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>
  
The objective of this project is to analyze Facebook post comments or feedback using sentiment analysis and filter only those entries that have positive sentiment. This helps in understanding user engagement and extracting constructive insights from social media data.
## Program:
```
!pip install pandas nltk matplotlib
```
```
import pandas as pd
import nltk
import matplotlib.pyplot as plt
```
```
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```
```
nltk.download('vader_lexicon')
```
```
file = "facebook_sentiment_dataset.csv"
df = pd.read_csv(file)
print(df.head())
```
```
print(df.shape)
print(df.columns.tolist())
```
```
print(df.isnull().sum())
```
```
df = df.dropna(subset=['Post'])
print("Number of posts:", len(df))
```
```
sid = SentimentIntensityAnalyzer()
```
```
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
```
```
df['Sentiment_Score'] = df['Post'].apply(
    lambda x: sid.polarity_scores(str(x))['compound']
)
print(df[['Post', 'Sentiment_Score']])
```
```
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
df['Sentiment'] = df['Sentiment_Score'].apply(classify_sentiment)
print(df[['Post', 'Sentiment_Score', 'Sentiment']])
```
```
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
```
```
positive_posts = df[df['Sentiment'] == 'Positive']

print("Positive posts:")
print(positive_posts[['Post_ID', 'Post', 'Sentiment']])
```
```
positive_posts[['Post_ID', 'Post', 'Sentiment']]
```
```
sentiment_counts.plot(kind='bar')

plt.title('Facebook Post Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.show()
```
```
positive_count = len(positive_posts)

plt.bar(['Positive Posts'], [positive_count])

plt.title('Positive Facebook Posts')
plt.ylabel('Number of Posts')
plt.show()
```
```
positive_posts.to_csv(
    'positive_facebook_posts.csv',
    index=False
)

print("Positive posts saved successfully!")
```
```
print("Total posts:", len(df))
print("Positive posts:", len(df[df['Sentiment'] == 'Positive']))
print("Negative posts:", len(df[df['Sentiment'] == 'Negative']))
print("Neutral posts:", len(df[df['Sentiment'] == 'Neutral']))
```


<H3>Output:</H3>

<img width="1108" height="744" alt="image" src="https://github.com/user-attachments/assets/2727c1e4-8d59-42db-8dd4-8c7c6421b6ee" />


<img width="730" height="522" alt="image" src="https://github.com/user-attachments/assets/1cdd771a-62fb-41b8-a47b-104946cdc687" />

<H3>Inference:</H3>
Through this project, I learned how to perform sentiment analysis on Facebook feedback using Python. I learned how to load and process a CSV dataset, apply VADER sentiment analysis, classify feedback as positive, negative, or neutral, and filter positive posts. I also learned how to visualize sentiment results and extract useful insights from social media data. Overall, the project improved my understanding of NLP, data preprocessing, Python programming, and data analysis.
