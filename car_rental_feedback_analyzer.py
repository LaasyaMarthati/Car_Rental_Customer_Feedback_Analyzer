
# Car Rental Customer Feedback Analyzer

# ------------------------------
# 1. Install Required Libraries

!pip uninstall transformers -y
!pip install transformers --upgrade

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------------------
# 2. Download Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------
# 3. Clean Text Function
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    tokens = text.lower().split()
    return ' '.join([word for word in tokens if word not in stop_words])

# ------------------------------
# 4. Load CSV File
df = pd.read_csv("car_rental_reviews.csv")
df['clean_review'] = df['review'].apply(clean_text)

# ------------------------------
# 5. Sentiment Analysis
classifier = pipeline("sentiment-analysis")
df['sentiment'] = classifier(df['review'].tolist())
df['sentiment'] = df['sentiment'].apply(lambda x: x['label'])

# ------------------------------
# 6. Extract Keywords
def extract_keywords(reviews, top_n=30):
    all_words = ' '.join(reviews).split()
    filtered = [word for word in all_words if word not in stop_words]
    freq_dist = nltk.FreqDist(filtered)
    return [word for word, _ in freq_dist.most_common(top_n)]

keywords = extract_keywords(df['clean_review'])

# ------------------------------
# 7. Extract Issues from Negative Reviews Only
def extract_issues(row, keywords):
    if row['sentiment'] == 'NEGATIVE':
        words = row['clean_review'].split()
        issues = [word for word in words if word in keywords]
        return ', '.join(issues) if issues else 'Unclear issue'
    return 'None'

df['issues'] = df.apply(lambda row: extract_issues(row, keywords), axis=1)

# ------------------------------
# 8. Display Results
for i, row in df.iterrows():
    print(f"\nReview {i+1}")
    print("Original :", row['review'])
    print("Sentiment:", row['sentiment'])
    print("Issues   :", row['issues'])

# ------------------------------
# 9. WordCloud Visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Frequent Keywords in Customer Feedback")
plt.show()