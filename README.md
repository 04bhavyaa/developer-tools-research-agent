# AI Research Agent for Developer Tools

This project is a sophisticated, multi-step AI agent designed to automate the process of researching and analyzing developer tools, frameworks, and services. Given a simple query (e.g., "open source alternatives to Firebase"), the agent performs web searches, scrapes relevant articles, extracts key information, and generates a final, structured analysis with recommendations.

The agent is built with a modern Python stack and features a user-friendly web interface created with Streamlit.

PS: This project was inspired from @TechWithTim on YT, I leveraged Gemini Key instead of Open AI.

Github Repo: https://github.com/techwithtim/Advanced-Research-Agent
## Features

* **Multi-Step Workflow:** Utilizes **LangGraph** to create a robust, multi-step workflow (extract tools -> research -> analyze).
* **Dynamic Web Scraping:** Employs **Firecrawl** to perform real-time web searches and scrape content from websites.
* **AI-Powered Analysis:** Leverages **Google's Gemini 2.5 Flash** model for intelligent data extraction and final analysis.
* **Structured Output:** Uses Pydantic models to ensure the data passed between steps is structured and validated.
* **Resilient Error Handling:** Includes a fallback mechanism to recover from failed steps and continue the research process.
* **Interactive UI:** Provides a clean and interactive web interface built with **Streamlit** for easy use.

## Tech Stack

* **Orchestration:** LangChain & LangGraph
* **LLM:** Google Gemini 2.5 Flash
* **Web Scraping/Search:** Firecrawl
* **Web UI:** Streamlit (WORKING ON IT!)
* **Package Management:** `uv`
* **Language:** Python 3.10+

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Initialize Project

Initialize the project with `uv`:

```bash
uv init .
```

### 3. Install Dependencies

Use `uv` to install all the required packages from `requirements.txt` or do this.

```bash
uv add firecrawl-py python-dotenv langchain-google-genai langchain langgraph pydantic python-dotenv
```

*(Streamlit integration coming soon)*

### 4. Set Up Environment Variables

Create a file named `.env` in the root of your project directory and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

## How to Run

Once the setup is complete, you can launch the Streamlit application.

```bash
uv run streamlit run main.py
```

Your web browser will automatically open to the application's URL (usually `http://localhost:8501`).

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── workflow.py         # Defines the core LangGraph agent and its steps
│   ├── firecrawl.py        # Service class for interacting with the Firecrawl API
│   ├── models.py           # Pydantic models for state and data structures
│   └── prompts.py          # Contains all prompts for the LLM
├── .env                    # Stores API keys and environment variables
├── .python-version         # Python version specification
├── main.py                 # The Streamlit application entry point
├── pyproject.toml          # Project configuration and dependencies
└── README.md               # You are here!
```

## Usage

1. Start the application using `uv run streamlit run main.py`
2. Enter your research query in the text input field
3. Click "Start Research" to begin the analysis
4. The agent will:
   - Extract relevant tools from your query
   - Perform web searches for each tool
   - Scrape and analyze content from multiple sources
   - Generate a comprehensive analysis with recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## API Keys

To use this application, you'll need API keys for:
- **Google Gemini API:** Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Firecrawl API:** Sign up at [Firecrawl](https://firecrawl.dev) to get your API key

## Troubleshooting

### Common Issues

1. **API Key Errors:** Make sure your `.env` file is in the root directory and contains valid API keys
2. **Virtual Environment Issues:** Ensure you've activated the virtual environment before installing packages
3. **Package Installation Problems:** Try using `pip` instead of `uv pip` if you encounter issues

### Support

If you encounter any issues, please check the following:
- Ensure all dependencies are installed correctly
- Verify that your API keys are valid and have sufficient quota
- Check that you're using Python 3.10 or higher
- Reach out at bhavyajha1404@gmail.com