# ARGUS

ARGUS is a local research assistant for text questions and image or video analysis. It can also capture a webcam or screen, route requests to vision or reasoning models, and add web search results when useful.

## Run locally

1. Install Python 3.9 or newer.
2. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Add your OpenRouter key to a root `.env` file:

   ```text
   OPENROUTER_API_KEY=your-key
   ```

4. Start ARGUS and open [http://127.0.0.1:5000](http://127.0.0.1:5000):

   ```powershell
   python app.py
   ```

The conversation is held in browser memory and clears on reload. Uploaded media and questions are sent to OpenRouter for analysis; review the selected model provider's data policy before sending sensitive material.
