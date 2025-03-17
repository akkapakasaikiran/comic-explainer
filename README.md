# AI Comic Explainers
This project uses AI to generate engaging comic explainers. The goal is to combine the power of large language models (LLMs) and generative image models to create comic strips that explain complex topics in a fun, educational way.

## Setup
To get started, you'll need to set up a virtual environment and configure some environment variables.

### Step 1: Create a Virtual Environment and Install Dependencies
```
python3 -m venv venv/comic-explainer
source venv/comic-explainer/bin/activate  
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables

We use Gemini and ElevenLabs in this project. You can create API keys by navigating to [link 1](https://ai.google.dev/gemini-api/docs/api-key) and [link 2](https://elevenlabs.io/app/settings/api-keys).
Create a .env file in the root directory and add the following lines:
```
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### Step 3: Run the Project
Once everything is set up, you can run the project:

```
python src/main.py
```
This will generate the comic explainer based on your input and store the results in the `comic_project` folder.


## Sample Output
You can find a sample output of the comic explainer in the `comic_project` folder.

