# No Limit Texas Hold'em Web App

A modern web-based interface for playing No Limit Texas Hold'em poker against AI opponents, built on top of the RLCard framework.

## Features

- **Modern Web Interface**: Beautiful, responsive UI with card animations
- **Real-time Game State**: Live updates of game state and opponent actions
- **Full Game Support**: All standard poker actions (Fold, Check/Call, Raise, All-In)
- **Visual Feedback**: Clear display of cards, chips, pot, and game stage
- **AI Opponent**: Play against an LLM-powered AI opponent that makes GTO-optimal decisions using tool calling
- **AI Coach Chat**: Always-visible conversational AI coach for strategy questions
- **Hand Analysis**: Automatic analysis after each completed hand with GTO-based grading

## Setup

### Prerequisites

- Python 3.6 or higher
- RLCard framework installed

### Installation

1. Install Python dependencies:
```bash
cd webapp
pip install -r requirements.txt
```

2. Make sure RLCard is installed. If not, install it:
```bash
cd ../rlcard
pip install -e .
```

3. Set up environment variables:
   - Create a `.env` file in the `webapp/` directory
   - Configure LLM provider and API keys:
     ```
     # LLM Provider Selection (choose one: 'openai' or 'openrouter')
     LLM_PROVIDER=openai

     # OpenAI Configuration (used when LLM_PROVIDER=openai)
     OPENAI_API_KEY=your_openai_api_key_here

     # OpenRouter Configuration (used when LLM_PROVIDER=openrouter)
     OPEN_ROUTER_KEY=your_openrouter_key_here
     ```
   - **LLM_PROVIDER**: Set to `openai` or `openrouter` to choose your preferred LLM provider
     - `openai`: Use OpenAI's API directly (more reliable, fewer models)
     - `openrouter`: Use OpenRouter (access to multiple models from different providers)
   - **OpenAI**: Get your key from https://platform.openai.com/api-keys
     - Default model: `gpt-4-turbo-preview`
   - **OpenRouter**: Get your key from https://openrouter.ai/keys
     - Default model: `openai/gpt-4-turbo`
     - See available models at https://openrouter.ai/models
   - **Note**: The `.env` file is gitignored and should not be committed. Both AI Coach and LLM Opponent will use the selected provider.

## Running the Application

1. Start the Flask server:
```bash
cd webapp
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Start a New Game**: Click the "New Game" button in the header
2. **Make Actions**: When it's your turn, click one of the action buttons:
   - **Fold**: Give up the hand
   - **Check/Call**: Match the current bet or check if no bet
   - **Raise ½ Pot**: Raise by half the pot size
   - **Raise Pot**: Raise by the full pot size
   - **All-In**: Bet all remaining chips
3. **View Game State**: The interface shows:
   - Your cards (always visible)
   - Community cards (revealed as game progresses)
   - Opponent cards (hidden until showdown)
   - Chip counts for both players
   - Current pot size
   - Game stage (Preflop, Flop, Turn, River, Showdown)

## API Endpoints

### Game Endpoints
- `GET /` - Serve the main web interface
- `POST /api/game/start` - Start a new game
- `GET /api/game/state` - Get current game state
- `POST /api/game/action` - Process a player action
- `POST /api/game/ai-turn` - Process AI opponent's turn

### Coach Endpoints
- `POST /api/coach/chat` - Chat with AI coach
  - Request: `{ "session_id": "string", "message": "string", "game_context": {...} }`
  - Response: `{ "response": "string", "timestamp": "ISO8601" }`
- `POST /api/coach/analyze-hand` - Analyze a completed hand
- `GET /api/coach/get-hand-history` - Get hand history for a session

## Architecture

### Backend (Flask)
- `app.py`: Main Flask application with game state management
- `GameManager`: Manages game instances and state
- `WebHumanAgent`: Human agent adapted for web interface

### Frontend
- `templates/index.html`: Main HTML structure
- `static/style.css`: Styling and layout
- `static/app.js`: Client-side game logic and API communication

## Differences from Desktop GUI

The web app version provides:
- Cross-platform compatibility (works on any device with a browser)
- No installation required for end users
- Modern, responsive design
- RESTful API architecture
- Easier deployment and distribution

## Troubleshooting

- **Port already in use**: Change the port in `app.py` (default: 5000)
- **RLCard import errors**: Make sure RLCard is properly installed
- **Game not starting**: Check browser console for JavaScript errors
- **Actions not working**: Verify the Flask server is running and accessible

## LLM-Powered Opponent

The opponent uses LLM (OpenAI/OpenRouter) with tool calling to make GTO-optimal decisions:

- **LLM-Powered Decisions**: All opponent decisions (preflop and postflop) are made by LLM using tool calling
- **Complete Context**: LLM receives full game context including cards, positions, action history, stacks, pot odds, and equity
- **GTO Strategy**: Decisions align with Game Theory Optimal principles
- **Robust Error Handling**: Automatic fallback to rules-based GTOAgent on LLM failures
- **Performance**: Optimized for < 3s preferred, < 5s acceptable decision latency
- **Pre-calculated Analysis**: Includes opponent range, hand equity, pot odds, and board texture analysis

**Note**: Requires a valid API key in your `.env` file (either `OPEN_ROUTER_KEY` or `OPENAI_API_KEY`). If no API key is available, the opponent automatically falls back to rules-based GTOAgent.

## AI Coach Chat

The AI Coach provides conversational strategy guidance using OpenAI GPT-4:

- **Always Visible**: Chat interface is always accessible (sidebar on desktop, bottom on mobile)
- **Context-Aware**: Automatically includes current game state when relevant
- **Educational**: Provides clear explanations of GTO strategy concepts
- **Question Types Supported**:
  - Strategy questions ("What's the optimal play here?")
  - Concept explanations ("What are pot odds?")
  - Hand analysis requests ("Analyze my last hand")
  - General poker education ("How do I improve my game?")

### Using the Chat

1. Type your question in the chat input field
2. Press Enter or click Send
3. The coach will respond with educational guidance
4. Conversation history is maintained during your session

**Note**: Requires a valid API key in your `.env` file (either `OPEN_ROUTER_KEY` or `OPENAI_API_KEY`).

## Future Enhancements

- WebSocket support for real-time updates
- Multiple AI difficulty levels
- Hand history and statistics
- Multiplayer support
- Tournament mode
- Custom bet amounts


