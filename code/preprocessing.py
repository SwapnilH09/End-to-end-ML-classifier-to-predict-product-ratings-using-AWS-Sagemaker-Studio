
import numpy as np
import pandas as pd
import string
from sklearn.utils import resample


base_dir = "/opt/ml/processing"

df = pd.read_csv(f"{base_dir}/input/Womens Clothing E-Commerce Reviews.csv")
df = df[df['Review Text'].notna()]  # drop rows without review text


def process_review(text):
    punctuation = string.punctuation
    review = text.lower()
    review = review.replace("\r\n", " ").replace("\n\n", " ")  # clean text of \r, \n and \n\n with whie space
    translator = str.maketrans("", "", punctuation)
    review = review.translate(translator)
    return review


# create columns to concatenate reviews and new labels
df['Complete_Review'] = df["Title"] + " " + df['Review Text']
df = df[df['Complete_Review'].notna()]  # drop rows with NA
# ratings 1 & 2 are mapped to negative reviews
# rating 3 & 4 are mapped to neutral
# rating 5 is mapped as positive
df['Label'] = df['Rating'].map({1:"negative", 2:"negative", 3:"none", 4:"none", 5:"positive"})
df = df.loc[df['Label'].isin(['negative', 'positive'])]  # using only neg and pos reviews
df['Review'] = df['Complete_Review'].astype(str).apply(process_review)  # applying process_review() on each review text
df['Processed'] = '__label__' + df['Label'].astype(str) + ' ' + df['Review']


# create train--test split
train, validation, test = np.split(df, [int(0.7 * len(df)), int(0.85 * len(df))])

# performing oversampling to deal with imbalance in dataset
positive = train.loc[train['Label']=='positive']
negative = train.loc[train['Label']=='negative']

# oversampling the minority classes
negative_oversample = resample(negative, replace=True, n_samples=len(positive))

# remake training sample 
train = pd.concat([positive, negative_oversample])

# create series dataset for Blazing Text input format
train = train['Processed']
validation = validation['Processed']
test = test['Processed']


# save datasets 
pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

print(f"Number of reviews in the training dataset: {train.shape[0]}")
print(f"Number of reviews in the validation set: {validation.shape[0]}")
