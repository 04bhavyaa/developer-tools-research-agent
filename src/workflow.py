from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import os

from pydantic import BaseModel

from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts


class Workflow:
    def __init__(self):
        """
        Initializes the workflow, setting up the Firecrawl service,
        the Gemini LLM, prompts, and compiling the graph.
        """
        self.firecrawl = FirecrawlService()
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """
        Builds the computational graph for the research agent using LangGraph.
        """
        # Define the state machine graph with the ResearchState schema
        graph = StateGraph(ResearchState)

        # Add the nodes for each step in the workflow
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)

        # Define the entry point and the edges that connect the nodes
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)  # The final step ends the graph execution

        # Compile the graph into a runnable workflow
        return graph.compile()

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node 1: Searches for articles and extracts a list of relevant tools.
        """
        print(f"--- Step 1: Finding articles about '{state.query}' ---")
        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=3)

        all_content = ""
        for result in search_results.data:
            url = result.get("url", "")
            if url:
                print(f"Scraping: {url}")
                scraped = self.firecrawl.scrape_company_pages(url)
                if scraped:
                    all_content += scraped.markdown[:1500] + "\n\n"

        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content))
        ]

        try:
            response = self.llm.invoke(messages)
            tool_names = [name.strip() for name in response.content.strip().split("\n") if name.strip()]
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(f"Error during tool extraction: {e}")
            return {"extracted_tools": []}

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis | dict | BaseModel:
        """
        Helper function: Uses the LLM's structured output to analyze a company's website.
        """
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)
        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]
        try:
            print(f"Analyzing content for: {company_name}")
            return structured_llm.invoke(messages)
        except Exception as e:
            print(f"Error during company analysis for {company_name}: {e}")
            return CompanyAnalysis(
                pricing_model="Unknown", is_open_source=None, tech_stack=[],
                description="Failed to analyze content.", api_available=None,
                language_support=[], integration_capabilities=[]
            )

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node 2: Researches each extracted tool to gather detailed information.
        """
        print("\n--- Step 2: Researching individual tools ---")
        extracted_tools = getattr(state, "extracted_tools", [])

        if not extracted_tools:
            print("No tools extracted. Falling back to direct search and title analysis.")
            search_results = self.firecrawl.search_companies(state.query, num_results=5)
            titles = [res.get("metadata", {}).get("title", "") for res in search_results.data]

            # Add a recovery mechanism by asking the LLM to extract names from titles
            recovery_messages = [
                SystemMessage(content="You are an expert at extracting company and product names from text."),
                HumanMessage(
                    content=f"From the following list of article titles, please extract the names of any companies or software products mentioned. Return only the names, one per line.\n\nTitles:\n- {'\n- '.join(titles)}")
            ]
            response = self.llm.invoke(recovery_messages)
            tool_names = [name.strip() for name in response.content.strip().split("\n") if name.strip()]
        else:
            tool_names = extracted_tools

        # Limit to researching the top 4 tools to keep it focused
        tool_names = tool_names[:4]

        print(f"Researching: {', '.join(tool_names)}")
        companies: List[CompanyInfo] = []
        for tool_name in tool_names:
            search_query = f"{tool_name} official site"
            tool_search_results = self.firecrawl.search_companies(search_query, num_results=1)

            if tool_search_results and tool_search_results.data:
                result = tool_search_results.data[0]
                url = result.get("url", "")
                print(f"Found site for {tool_name}: {url}")

                company = CompanyInfo(name=tool_name, description=result.get("markdown", ""), website=url)
                scraped = self.firecrawl.scrape_company_pages(url)
                if scraped:
                    analysis = self._analyze_company_content(company.name, scraped.markdown)
                    company.pricing_model = analysis.pricing_model
                    company.is_open_source = analysis.is_open_source
                    company.tech_stack = analysis.tech_stack
                    company.description = analysis.description
                    company.api_available = analysis.api_available
                    company.language_support = analysis.language_support
                    company.integration_capabilities = analysis.integration_capabilities
                companies.append(company)
            else:
                print(f"Could not find an official site for {tool_name}")
        return {"companies": companies}

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node 3: Generates a final analysis and recommendation based on all gathered data.
        """
        print("\n--- Step 3: Generating final recommendations ---")

        # Create a formatted string for each company to avoid serialization issues.
        company_summaries = []
        for company in state.companies:
            summary = (
                f"Company: {company.name}\n"
                f"Website: {company.website}\n"
                f"Description: {company.description}\n"
                f"Pricing Model: {company.pricing_model}\n"
                f"Open Source: {company.is_open_source}\n"
                f"API Available: {company.api_available}\n"
                f"Tech Stack: {', '.join(company.tech_stack)}\n"
                f"Language Support: {', '.join(company.language_support)}\n"
                f"Integration Capabilities: {', '.join(company.integration_capabilities)}"
            )
            company_summaries.append(summary)

        company_data = "\n\n---\n\n".join(company_summaries)

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATION_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data))
        ]
        response = self.llm.invoke(messages)
        print("Analysis complete.")
        return {"analysis": response.content}

    def run(self, query: str) -> ResearchState:
        """
        Executes the entire research workflow for a given query.
        """
        # Define the initial state for the graph
        initial_state = {"query": query}

        # Invoke the compiled graph with the initial state
        # The stream() method can also be used for real-time updates
        final_state = self.workflow.invoke(initial_state)

        # Return the final state as a Pydantic model
        return ResearchState(**final_state)
