import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter

import seaborn as sns

df= pd.read_csv("C:/Users/yakis/Documents/GitHub/IUPEProject/IUPEProject/Phishing_Email.csv")
#print(df.isna().sum()) #shows us what values are NA
df = df.dropna() # drops the NA values
#print(df.shape())

#exploring the data set
email_type_counts = df['Email Type'].value_counts() #counts the different type of emails
print(email_type_counts)
#Distribution of Phishing to safe
unique_email_types = email_type_counts.index.tolist()
color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',}
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]
plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Exploring the most used words in phishing emails
phishing_emails = df[df['Email Type'] == 'Phishing Email']
texts = phishing_emails['Email Text'].astype(str)
#nltk.download('punkt')
#nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

words = []

for text in texts:
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    words.extend(filtered)

word_freq = Counter(words)
common_words = word_freq.most_common(20)
print(common_words)
phishing_df = df[df['Email Type'] == 'Phishing Email'].copy()
phishing_df['Email Text'] = phishing_df['Email Text'].astype(str)

safe_emails = df[df['Email Type'] == 'Safe Email']
safeText = safe_emails['Email Text'].astype(str)

safewords = []

for text in safeText:
    tokens = word_tokenize(text.lower())
    filteredSafe = [word for word in tokens if word.isalpha() and word not in stop_words]
    safewords.extend(filteredSafe)

word_freq_safe = Counter(safewords)
common_Safewords = word_freq_safe.most_common(20)
print(common_Safewords)
"""
for word, count in common_words:
    print(f"{word}: {count}")

    weird_matches = phishing_df[phishing_df['Email Text'].str.contains('â', na=False)]

for i, text in enumerate(weird_matches['Email Text'].head(5), 1):
    print(f"{i}. {text}\n")
"""
"""
# Convert to DataFrame for plotting
freq_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='word', data=freq_df, palette='viridis')
plt.title('Top 20 Most Common Words in Phishing Emails')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.tight_layout()
plt.show()"""