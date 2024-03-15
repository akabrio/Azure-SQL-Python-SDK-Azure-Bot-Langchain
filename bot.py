# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from common.prompts import WELCOME_MESSAGE
from utils import SQLSearchAgent


class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    def __init__(self):  
        super(MyBot, self).__init__()
          
        self.sql_agent = SQLSearchAgent()  

          
    async def on_message_activity(self, turn_context: TurnContext):  
        query = turn_context.activity.text.strip().lower()  
        response = ""  

        # Check for specific greetings or questions  
        if query.startswith("hi") or query.startswith("hello"):  
             response = "Hello! My name is Jarvis, a smart virtual assistant designed to assist you, How can I assist You?"  
        elif "who are you" in query:  
                response = "My name is Jarvis, a smart virtual assistant designed to assist you."  
        elif query.startswith("my name is "):  
            name = query[11:].strip().capitalize()  # Extracting the name from the query
            response = f"Hello, {name}! My name is Jarvis, a smart virtual assistant designed to assist you, How can I assist You?"  
        elif "bye" in query:  
            response = "If you have any more questions or need further assistance in the future, feel free to reach out. Have a great day!"
        else:  
                # Process other queries using the existing logic  
                response = await self._run(query)  
            
        await turn_context.send_activity(response) 
             

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE)

    async def _run(self, query: str) -> str:
        # Call the _run method of SQLSearchAgent instance
        return self.sql_agent._run(query)