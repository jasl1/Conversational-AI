# Customer Service Chatbot

### Introduction
In this project, we will develop a conversational AI assistant (chatbot) to handle customer inquiries and provide support for a hypothetical e-commerce company. The chatbot will be capable of understanding natural language queries, retrieving relevant information, and generating appropriate responses.

### Prerequisites
Before starting this project, ensure that you have the following prerequisites installed:
* Python 3.7 or higher
* PyTorch
* Transformers (Hugging Face library)
* RASA (Open-source conversational AI framework)
* spaCy (Natural Language Processing library)
* SQLite (or any other database engine)

### Step 1: Load the Dataset
First, we need to load the Stanford Conversational AI dataset, which contains conversations between customers and agents in various domains like banking, travel, and shopping.
```python
import pandas as pd

# Load the dataset
data = pd.read_json('https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/data/stanford_convertional_ai/stanford_convertional_ai.json')
```

### Step 2: Preprocess the Data
Next, we'll preprocess the data by separating the conversations into user utterances and agent responses.

```python
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

```

###  Step 3: Build the NLU Model
We'll use the RASA framework to build the Natural Language Understanding (NLU) model for intent classification and entity extraction.
```python
from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer

# Create training data in RASA format
training_data = load_data("nlu_data.yml")

# Define the NLU pipeline
nlu_config = RasaNLUModelConfig({"pipeline": [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {"name": "SpacyFeaturizer"},
    {"name": "RegexFeaturizer"},
    {"name": "CRFEntityExtractor"},
    {"name": "EntitySynonymMapper"},
    {"name": "SklearnIntentClassifier"}
]})

# Train the NLU model
trainer = Trainer(nlu_config)
interpreter = trainer.train(training_data)

```

### Step 4: Build the Dialogue Management Model
Next, we'll define the dialogue management logic using RASA's domain file and story files.
```python
from rasa.core.agent import Agent
from rasa.core.train import train_dialogue_model

# Load the domain and stories
domain = "domain.yml"
stories = "data/stories.md"

# Train the dialogue management model
agent = train_dialogue_model(domain, stories)

```

### Step 5: Build the Action Server
We'll create an action server to handle custom actions and integrate with external systems (e.g., databases, APIs).

```python
from rasa_sdk import Action
import sqlite3

class ActionCheckOrderStatus(Action):
    def name(self):
        return "action_check_order_status"

    def run(self, dispatcher, tracker, domain):
        # Get the order ID from the user's message
        order_id = tracker.get_slot("order_id")

        # Connect to the database and retrieve order status
        conn = sqlite3.connect("orders.db")
        c = conn.cursor()
        c.execute("SELECT status FROM orders WHERE id=?", (order_id,))
        status = c.fetchone()[0]
        conn.close()

        # Send the order status to the user
        response = f"The status of your order {order_id} is: {status}"
        dispatcher.utter_message(response)

        return []

```

###  Step 6: Run the Chatbot
Finally, we can run the chatbot by combining the NLU model, dialogue management model, and action server.
```python
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter

# Load the trained models
interpreter = RasaNLUInterpreter("models/nlu")
agent = Agent.load("models/dialogue", interpreter=interpreter)

# Run the chatbot
while True:
    user_input = input("User: ")
    responses = agent.handle_text(user_input)
    for response in responses:
        print("Chatbot:", response.get("text"))

```

This example demonstrates how to build a conversational AI chatbot using the Stanford Conversational AI dataset, RASA framework, and Python. It covers the key steps of loading and preprocessing the dataset, training the NLU and dialogue management models, building custom actions, and running the chatbot.
