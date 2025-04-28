import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_path = "/projectnb/cs505aw/projects/DepressionEmo/revised_dataset/chat_gpt_annotated/combined_labeled_many_emotions.json"
original_df = pd.read_json(data_path, lines=True)

### transforming input data to more usable form
reduced_df = pd.DataFrame(columns= ['hopelessness', 'humiliation', 'anger', 'guilt', 'substance abuse', 'loneliness', 'suicide intent']) # one-hot columns
for index, row in original_df.iterrows():
    emo_list = row['emotions']
    reduced_df.loc[index] = 0
    # one hot encode the emotions
    for e in reduced_df.columns:
        if e in emo_list:
            reduced_df.loc[index, e] = 1

print("reduced dataframe:\n", reduced_df.head())
print("Percent of data with suicide intent label:", sum(reduced_df['suicide intent'])/sum(reduced_df['suicide intent'] != None))

### actual modeling section
target = np.array(reduced_df['suicide intent'])
input_data = np.array(reduced_df.drop(columns=['suicide intent']))
print(f"target shape: {target.shape} and input shape: {input_data.shape}")
# get 60 20 20 split between train, val, and test sets
train_X, test_X, train_y, test_y = train_test_split(input_data, target, test_size=0.2)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25)

model = LogisticRegression()
model.fit(train_X, train_y)

# results!
print(f"model input:\n{train_X[:5]}")
print(f"model ground truth:\n{train_y[:5]}\n")
print(f"model probability predictions: \n{model.predict_proba(train_X)[:5]}\n") # decison threshold is 0.5
print(f"model decision function: \n{model.decision_function(train_X)[:5]}\n") # decision threshold is 0
print(f"model validation performance: {model.score(val_X, val_y)}")
print(f"model test performance: {model.score(test_X, test_y)}\n")

# some analysis of results:
middle_total = 0
for row in model.predict_proba(train_X):
    if row[0] > 0.25 and row[0] < 0.75:
        middle_total += 1
print("Count of probabilities in middle (0.25-0.75):", middle_total)
