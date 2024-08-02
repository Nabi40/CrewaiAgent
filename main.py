from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks
from file_io import save_markdown

from dotenv import load_dotenv
load_dotenv()

def main():
    try:
        # Initialize the agents and tasks
        agents = AINewsLetterAgents()
        tasks = AINewsLetterTasks()

        # Initialize the OpenAI GPT-4 language model
        OpenAIGPT4 = ChatOpenAI(
            model="gpt-3.5-turbo"
        )

        # Get input question from the user
        question = input("Enter your question: ")

        # Instantiate the agents
        editor = agents.editor_agent()
        news_fetcher = agents.news_fetcher_agent()
        news_analyzer = agents.news_analyzer_agent()
        newsletter_compiler = agents.newsletter_compiler_agent()

        # Instantiate the tasks with the input question
        fetch_news_task = tasks.fetch_news_task(news_fetcher, question)
        analyze_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task], question)
        compile_newsletter_task = tasks.compile_newsletter_task(
            newsletter_compiler, [analyze_news_task], save_markdown, question)

        # Form the crew
        crew = Crew(
            agents=[editor, news_fetcher, news_analyzer, newsletter_compiler],
            tasks=[fetch_news_task, analyze_news_task, compile_newsletter_task],
            process=Process.hierarchical,
            manager_llm=OpenAIGPT4,
            verbose=True  # Change this to True or False
        )

        # Kick off the crew's work
        results = crew.kickoff()

        # Print the results
        print("Crew Work Results:")
        print(results)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
