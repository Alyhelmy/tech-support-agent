# Tech Support Agent

An AI-powered tech support assistant that combines the power of Llama2 with a knowledge base of technical solutions. The system uses Retrieval-Augmented Generation (RAG) and FAISS for efficient similarity search.

## Features

- **AI Chat Interface**: Real-time conversation with Llama2 AI model
- **Knowledge Base Integration**: Access to a comprehensive database of technical solutions
- **RAG Implementation**: Combines AI responses with relevant knowledge base articles
- **FAISS Vector Search**: Efficient similarity search for finding relevant solutions
- **Modern GUI**: User-friendly interface with chat and knowledge base tabs
- **Real-time Updates**: Automatic knowledge base updates and solution retrieval

## Requirements

- Python 3.8 or higher
- Ollama (for running Llama2 locally)
- Required Python packages (see requirements.txt):
  - torch>=2.1.0
  - langchain
  - faiss-cpu
  - sentence-transformers
  - tkinter (usually comes with Python)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alyhelmy/tech-support-agent.git
   cd tech-support-agent
   ```

2. Install Ollama:
   - Follow instructions at [Ollama's website](https://ollama.ai)
   - Pull the Llama2 model:
     ```bash
     ollama pull llama2
     ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python tech_support_gui.py
   ```

2. Using the Chat Interface:
   - Type your question in the chat input
   - The AI will respond with relevant information
   - If knowledge base articles are found, you'll be notified to check the Knowledge Base tab

3. Using the Knowledge Base:
   - Switch to the Knowledge Base tab
   - Search for specific solutions
   - View results with relevance scores
   - Click on any result to see the full solution

## Project Structure

```
tech-support-agent/
├── tech_support_agent.py    # Core agent implementation
├── tech_support_gui.py      # GUI implementation
├── requirements.txt         # Python dependencies
├── knowledge_base/         # Technical solutions database
└── .gitignore             # Git ignore rules
```

## Features in Detail

### AI Chat Interface
- Real-time conversation with Llama2
- Message history with timestamps
- Color-coded messages for better readability
- Automatic scrolling to latest messages

### Knowledge Base Integration
- Comprehensive database of technical solutions
- Efficient search using FAISS
- Relevance scoring for search results
- Split-pane view for browsing solutions

### RAG Implementation
- Combines AI responses with knowledge base articles
- Contextual understanding of queries
- Improved response accuracy
- Automatic knowledge base updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 