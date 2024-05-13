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
