# IAS Program Assistant

IAS Program Assistant is a command-line study assistant for students in the University of Hamburg's Intelligent Adaptive Systems (IAS) program. It answers questions using saved university documents and can search official University of Hamburg webpages when current information is needed.

## Features

- Retrieves information from local IAS and University of Hamburg documents.
- Searches only official `uni-hamburg.de` webpages for current information.
- Shows tool calls and tool results while the agent is working.
- Remembers earlier messages during the current chat session.
- Avoids invented application records or other fictional university data.

## Project structure

```text
IAS_Program_Assistant/
├── agent.py             # Chat loop and agent configuration
├── rag.py               # Document loading, chunking, and vector indexing
├── tools.py             # Local knowledge and live web-search tools
├── Data/                # English text used by the RAG system
├── SourceDocuments/     # Original PDFs, presentations, and webpage snapshots
├── .env                 # Local API key; excluded from Git
└── README.md
```

The files in `Data/` are the versions searched by the RAG system. The files in `SourceDocuments/` preserve the original source material. German text sources in `Data/` have been translated into English and clearly marked as unofficial translations; the linked official source remains authoritative.

## Setup

Python 3.11 or newer is recommended.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install llama-index openai python-dotenv
   ```

3. Create a `.env` file in the project directory:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the assistant:

   ```bash
   python agent.py
   ```

5. Enter a question. Type `exit` or `quit` to end the session.

Example questions:

```text
How many credits does the IAS thesis have?
What are the current course registration dates?
Which modules are part of the IAS program?
```

## How it works

When the application starts, `rag.py` reads every `.txt` file in `Data/`, splits the documents into smaller sections, creates OpenAI embeddings, and builds an in-memory vector index.

For each user message, `agent.py` starts an agent run:

```python
handler = agent.run(user_msg=question, ctx=context)
```

The `handler` represents that one running request. The program uses it to stream intermediate tool events and then waits for the final answer:

```python
async for event in handler.stream_events():
    # Display tool calls and results.

response = await handler
```

The language model selects a tool from its name, description, expected arguments, system prompt, and the user's question:

- `university_knowledge` searches the saved documents for stable program information.
- `search_university_web` searches official University of Hamburg pages for current or missing information.

The web-search tool is limited to `uni-hamburg.de`. Examination rules can depend on the student's cohort, so the agent is instructed to confirm the cohort before choosing a regulation version.

## Adding or updating sources

1. Obtain the information from an official University of Hamburg source.
2. Preserve the original file or webpage snapshot in `SourceDocuments/`.
3. Add a clean English `.txt` version to `Data/`.
4. Include the source URL, retrieval date, original language, and any freshness or applicability warning at the top of the text file.
5. Restart the application so the in-memory index is rebuilt.

Only `.txt` files in `Data/` are indexed. PDFs, PowerPoint files, and HTML snapshots in `SourceDocuments/` are retained as references but are not searched directly.

## Notes

- Starting the application creates embeddings again because the vector index is not persisted yet.
- Agent responses and live search use the OpenAI API and may incur API usage costs.
- University procedures and deadlines can change. Check linked official sources before relying on time-sensitive information.
