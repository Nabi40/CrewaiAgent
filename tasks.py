from datetime import datetime
from crewai import Task

class AINewsLetterTasks():
    def fetch_news_task(self, agent, question):
        return Task(
            description=f'Fetch news related to "{question}" from "https://en.wikipedia.org/wiki/Bangladesh" {datetime.now()}.',
            agent=agent,
            async_execution=False,
            expected_output=f"""Bangladesh: 
                [
                    {{ 'title': '{question}', 
                    'url': 'https://example.com/story1', 
                    'summary': '{question}'
                    }}, 
                    {{...}}
                ]
            """
        )

    def analyze_news_task(self, agent, context, question):
        return Task(
            description=f'Analyze each news story related to "{question}" and ensure there are at least 5 well-formatted articles',
            agent=agent,
            async_execution=False,
            context=context,
            expected_output=f"""A markdown-formatted analysis for each news story, including a rundown, detailed bullet points, 
                and a "Why it matters" section. There should be at least 5 articles, each following the proper format.
                Example Output: 
                '## {question} \n\n
                **The Rundown:
                ** {question}\n\n
                **The details:**\n\n
                - Microsoft\'s Copilot spot showcased its AI assistant...\n\n
                **Why it matters:** {question} \n\n'
            """
        )

    def compile_newsletter_task(self, agent, context, callback_function, question):
        return Task(
            description=f'Compile the newsletter related to "{question}"',
            agent=agent,
            context=context,
            async_execution=False,
            expected_output=f"""A complete newsletter in markdown format, with a consistent style and layout.
                Example Output: 
                '# {question}:\\n\\n
                - {question}\\n
                
                ## {question} \\n\\n
                **The Rundown:** {question} \\n\\n
                **The details:**...\\n\\n
                **Why it matters:**...\\n\\n
                ## {question} \\n\\n
                **The Rundown:** {question}...\\n\\n'
                **The details:**...\\n\\n
                **Why it matters:**...\\n\\n
            """,
            callback=callback_function
        )
