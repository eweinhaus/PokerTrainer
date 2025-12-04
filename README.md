# PokerTrainer

An AI-powered poker coaching system built on top of the RLCard framework. This web application provides real-time strategic analysis and educational feedback for No Limit Texas Hold'em poker players.

## Features

- **Interactive Poker Game**: Play No Limit Texas Hold'em against AI opponents
- **Real-time Strategy Evaluation**: Get instant A-F grading on your decisions with detailed explanations
- **AI Coach Chatbot**: Conversational interface for strategic questions and advice
- **GTO-Based Analysis**: Evaluations based on Game Theory Optimal principles
- **Hand Analysis**: Comprehensive post-hand review and improvement suggestions
- **Educational Feedback**: Learn poker strategy through interactive coaching

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: OpenAI/Anthropic APIs for LLM integration
- **Game Engine**: RLCard framework
- **Poker Logic**: Custom GTO-based evaluation engine

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PokerTrainer.git
cd PokerTrainer
```

2. Install dependencies:
```bash
cd webapp
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
./run.sh
```

5. Open your browser to `http://localhost:5000`

## Project Structure

```
PokerTrainer/
├── webapp/                 # Main Flask application
│   ├── app.py             # Main application entry point
│   ├── coach/             # AI coaching system
│   │   ├── strategy_evaluator.py  # GTO-based evaluation
│   │   ├── chatbot_coach.py       # LLM-powered coach
│   │   ├── gto_agent.py          # GTO strategy agent
│   │   └── ...
│   ├── static/            # CSS, JS assets
│   ├── templates/         # HTML templates
│   └── tests/             # Comprehensive test suite
├── rlcard/                # RLCard framework (submodule)
├── test_*.py             # Development test scripts
└── README.md
```

## Development

This project follows a phased development approach with comprehensive testing at each stage. See the test suite for detailed validation of all features.

## License

This project builds upon the RLCard framework. See individual component licenses for details.
