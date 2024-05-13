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
