from rasa.core.agent import Agent
from rasa.core.train import train_dialogue_model

# Load the domain and stories
domain = "domain.yml"
stories = "data/stories.md"

# Train the dialogue management model
agent = train_dialogue_model(domain, stories)
