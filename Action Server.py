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
