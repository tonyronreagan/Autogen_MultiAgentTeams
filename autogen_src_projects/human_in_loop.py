#Human in the loop
#feedback during a run 


from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, ExternalTermination, TextMentionTermination
from dotenv import load_dotenv
import os 
import asyncio

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ.pop("SSL_CERT_FILE", None)
 
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=api_key 
    )

myPA_agent = AssistantAgent(
    name="custom_PA",
    model_client=model_client,
    system_message="You are helpful assistant. You can answer questions about various topics. "
)

user_proxy_agent = UserProxyAgent(
    name='UserProxy',
    description="You are a user proxy agent that represents the user in the conversation. ",
    input_func= input
)

text_terminationCondition = TextMentionTermination(['APPROVED', 'DONE', 'YES','OK','GOTCHA'])
# #text_terminationCondition = TextMentionTermination(
#     ['APPROVED', 'DONE', 'YES', 'OK', 'GOTCHA'],
#     case_insensitive=True
# )

team = RoundRobinGroupChat(
    participants=[myPA_agent, user_proxy_agent],
    termination_condition=text_terminationCondition  # Set the termination condition
)

stream = team.run_stream(task="Can you help with a short plan for Agentic AI interview preparation?")

async def main():
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())