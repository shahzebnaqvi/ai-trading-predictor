
# Project Setup and Usage

## Setup Complete!

Follow these steps to set up and use the project:

1. **Train the Model**
    ```bash
    cd backend
    source venv/bin/activate
    python train_model.py
    python3 train_model.py X:BTCUSD --days 30 --interval minute

2. **Run the Server**
    ```bash
    python prediction_server.py

3. **Load the Chrome Extension**
    ```bash
    Open Chrome and navigate to chrome://extensions.
    Enable Developer mode (toggle in the top-right corner).
    Click Load unpacked and select the extension directory.

4. **Use the Extension**
    ```bash    
    Open TradingView.
    Click the extension icon in the Chrome toolbar to activate it.


### Notes
- Ensure you have all dependencies installed in the `backend` directory before training the model or running the server.
- The Chrome extension directory must contain a valid `manifest.json` file.
- If you encounter issues, verify that the virtual environment (`venv`) is properly set up and activated.

This Markdown content is concise, follows standard formatting, and can be directly added to your `README.md` file. Let me know if you need additional sections (e.g., prerequisites, troubleshooting) or further customization!