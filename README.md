# Tech Support Agent

A Python-based technical support agent that uses natural language processing to help diagnose and solve technical issues.

## Features

- Natural language query processing
- Fuzzy matching for better search results
- GUI interface for easy interaction
- Knowledge base management
- Real-time search results

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repository-url>
cd tech-support-agent
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python tech_support_gui.py
```

## Project Structure

- `tech_support_gui.py` - Main GUI application
- `tech_support_agent.py` - Core agent functionality
- `knowledge_store.py` - Database management
- `knowledge_base/` - Directory containing technical support documentation
- `requirements.txt` - Python dependencies

## Usage

1. Launch the application
2. Enter your technical issue in the search box
3. Click "Search" or press Enter
4. Select from the matching solutions if multiple are found
5. Use the "Refresh Knowledge Base" button to update with new files
6. Use "Show Database Info" to check the knowledge base status

## Requirements

- Python 3.7 or higher
- Dependencies listed in requirements.txt

## License

[Your chosen license] 