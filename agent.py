import os
import sys
import asyncio
from dotenv import load_dotenv

#agent libraries
from google.adk.agents import LlmAgent,SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Part,UserContent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner

load_dotenv() 

project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

research_agent = LlmAgent(
    name="Researcher",
    model="gemini-2.5-flash",
    instruction="""
    if 'greeted' is not set in state,greet the user with "Hello ! I am your assitant . How can i help you".
    Then conduct a thorough web research on the subject given in the input.
    Gather key facts,recent information, and relevant details about the topic.
    Summarize your research findings in a comprehensive but organized manner.
    This research will be used by other agents to write content, so be thorough and factual

    """,
    output_key = "research",
    tools = [google_search]
)

generator_agent = LlmAgent(
    name="DraftWriter",
    model="gemini-2.5-flash",
    instruction="""
    using the research findings stored in state["research"], write a short,factual paragraph about this subject.
    dont make up your facts. Be concise and clear. your output should be plain text only.
    """,
    output_key = "draft_text"
)

reviewer_agent=LlmAgent(
    name="Critic",
    model="gemini-2.5-flash",
    instruction="""
    Analyze the paragraph stored in state['draft_text'].
    Also reference the research in state['research'] to check factual accuracy.
    Always give a detailed critique, even if the text is factually correct,comment on style,clarity,possible improvements or anything that could make it better. if there are factual problem. mention those.Output a short critique,not just 'valid'.
    """,
    output_key = "critique"
)


revision_agent = LlmAgent(
    name="Rewriter",
    model="gemini-2.5-flash",
    instruction="""
Your goal is to revise the paragraph in state['draft_text'] based on the feedback in state['critique'].
You can also reference the research in state['research'] to ensure accuracy and completeness.
Output only the improved paragraph, rewritten as needed.
""",
    output_key="revised_text",
)

def greet_on_first_message(callback_context:CallbackContext):
    if not callback_context.state.get("greeted"):
        callback_context.state["greeted"] = True
    return None


sequential_root_agent = SequentialAgent(
    name="ResearchWriteCritiqueRewrite",
    before_agent_callback=greet_on_first_message,
    sub_agents=[research_agent,generator_agent,reviewer_agent,revision_agent]
)

root_agent = sequential_root_agent


