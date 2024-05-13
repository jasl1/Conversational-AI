from sklearn.model_selection import train_test_split

# Separate user utterances and agent responses
utterances = []
responses = []

for conversation in data['conversations']:
    for utterance, response in zip(conversation['utterances'], conversation['responses']):
        utterances.append(utterance)
        responses.append(response)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(utterances, responses, test_size=0.2, random_state=42)
