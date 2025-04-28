# find most common emotions that show up with suicide intent
# find most common set of emotions that show up with suicide intent

# find what pops up most without suicide intent? same 2 things

import pandas as pd
import numpy as np 
import collections
import pprint

# gather information from just training set, can verify with test set
df = pd.read_json('dataset/train.json', lines=True) 

set_count_with_intent = collections.defaultdict(int)
indiv_count_with_intent = collections.defaultdict(int)
set_count_without_intent = collections.defaultdict(int)
indiv_count_without_intent = collections.defaultdict(int)

for index, row in df.iterrows():
    emotions = row['emotions']
    if 'suicide intent' in emotions:
        other_emotions = [e for e in emotions if e != 'suicide intent']
        set_count_with_intent[tuple(sorted(other_emotions))] += 1
        for e in other_emotions:
            indiv_count_with_intent[e] += 1
    else:
        other_emotions = emotions
        set_count_without_intent[tuple(sorted(other_emotions))] += 1
        for e in other_emotions:
            indiv_count_without_intent[e] += 1

# calculating individual percentage with suicide intent
percent_with_suicide_intent = {}
for e, val_with in indiv_count_with_intent.items():
    val_without = indiv_count_without_intent[e]
    percent_with_suicide_intent[e] = val_with / (val_with + val_without)

# calculating percentage with suicide intent for sets
percent_with_suicide_intent_sets = {}
for e, val_with in set_count_with_intent.items():
    val_without = set_count_without_intent[e]
    # percentage with suicide intent for individual emotions
    percent_with_suicide_intent_sets[e] = val_with / (val_with + val_without)
print("length of percentages of suicide intent with sets", len(percent_with_suicide_intent))
sorted_percent_with_suicide_intent_sets = list(sorted(percent_with_suicide_intent_sets.items(), key=lambda x: x[1], reverse=True))[:10]

print("percent with suicide intent")
pprint.pprint(sorted(percent_with_suicide_intent.items(), key=lambda x: x[1], reverse=True))

print("\nTop 10 highest percentages with suicide intent (sets)")
pprint.pprint(sorted_percent_with_suicide_intent_sets)