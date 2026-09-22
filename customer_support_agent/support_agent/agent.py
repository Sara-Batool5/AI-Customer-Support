"""
Single-agent CrewAI setup for the customer support assistant.

The agent has three tools:
  - search_company_knowledge  (FAISS + chunks.json)
  - lookup_order               (Excel "database")
  - escalate_to_human           (writes a pending ticket)

Chat "memory" is handled simply: app.py keeps the full conversation in
st.session_state and passes it into the Task description on every turn,
so the agent always sees the whole conversation so far. This avoids
needing extra memory infrastructure / API keys for a beginner project.
"""

from crewai import Agent, Task, Crew, Process, LLM

from tools import KnowledgeSearchTool, OrderLookupTool, EscalateToHumanTool

# Gemini 3.5 Flash-Lite: fast, low-cost, good fit for a support chat agent.
GEMINI_MODEL = "gemini/gemini-3.5-flash-lite"


def build_llm(api_key: str) -> LLM:
    return LLM(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0.3,
    )


def build_agent(api_key: str) -> Agent:
    llm = build_llm(api_key)
    return Agent(
        role="Customer Support Specialist",
        goal=(
            "Resolve customer questions using the company knowledge base and the order "
            "lookup tool. Escalate to a human whenever you cannot fully resolve an issue, "
            "or whenever the customer asks for a human."
        ),
        backstory=(
            "You are a friendly, efficient customer support agent for an online tech "
            "accessories store serving customers across Pakistan. You always try the "
            "search_company_knowledge tool for policy/product questions and the "
            "lookup_order tool for order-status questions before answering. You never "
            "invent order details or policies that the tools did not return. If a tool "
            "finds nothing relevant, say so honestly rather than guessing. If you still "
            "can't help, or the customer asks for a human, use escalate_to_human, then "
            "clearly tell the customer their request has been escalated."
        ),
        tools=[KnowledgeSearchTool(), OrderLookupTool(), EscalateToHumanTool()],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_task(agent: Agent, conversation_history: str, user_message: str) -> Task:
    description = (
        "You are continuing an ongoing customer support chat. Conversation so far "
        "(oldest to newest):\n\n"
        f"{conversation_history}\n\n"
        f'The customer\'s newest message is: "{user_message}"\n\n'
        "Respond only to this newest message, using the earlier conversation as context. "
        "Use your tools whenever the answer depends on company knowledge or order data — "
        "never guess at policy or order details. If you cannot resolve the issue, or the "
        "customer asks for a human, call escalate_to_human and then tell the customer "
        "plainly that their request has been escalated to a human."
    )
    return Task(
        description=description,
        expected_output="A clear, helpful, concise reply to the customer's newest message.",
        agent=agent,
    )


def get_response(api_key: str, conversation_history: str, user_message: str) -> str:
    """Runs one turn of the support conversation and returns the agent's reply text."""
    agent = build_agent(api_key)
    task = build_task(agent, conversation_history, user_message)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)
