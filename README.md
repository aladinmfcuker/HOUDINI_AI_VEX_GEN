# HOUDINI_AI_VEX_GEN
hoidini ai vex gen by ret.ouchs(insta)\

Houdini AI VEX Generator Shelf Tool
Version: 1.1
Date: 2024-06-12
Description: Generates VEX code using selected AI provider (Anthropic, Gemini, or OpenAI).
             Applies generated code to a new or selected Attribute Wrangle node.

Requirements:
- Houdini 18.0+ (for PySide2)
- RUN THIS: *Houdini>shell>hython -m pip install google-generativeai openai anthropic*
- Python 3.x (standard with recent Houdini versions)
- anthropic (>= 0.20.0), google-generativeai, and openai (>= 1.0.0) Python packages
- API keys for the services you wish to use

Installation:
1. Save this file (e.g., ai_vex_generator.py) to your Houdini scripts directory
   (e.g., ~/houdiniX.Y/scripts).
2. Create a new Tool on a Houdini shelf.
3. In the Tool editor's 'Script' tab, set the script content to:
   import ai_vex_generator
   ai_vex_generator.run_ai_vex_tool()
4. Configure your API keys as described below.
5. Restart Houdini or update the Python path if needed.

API Key Configuration:
---------------------
Option 1 (Recommended - Secure): Set Environment Variables
  - Set ANTHROPIC_API_KEY for Anthropic Claude
  - Set GOOGLE_API_KEY for Google Gemini
  - Set OPENAI_API_KEY for OpenAI
  How to set environment variables depends on your OS (e.g., .bashrc, .zshrc on Linux/macOS,
  System Properties -> Environment Variables on Windows). Houdini needs to be launched
  from a shell where these variables are defined.

Option 2 (Less Secure - Use with Caution): Paste keys directly below.
  WARNING: Hardcoding keys in scripts is a security risk. Only use this if you
           understand the implications and cannot use environment variables.
           Avoid sharing scripts with hardcoded keys.

Replace your api keys in ai_vex_generator.py (easy method but unsecure)
