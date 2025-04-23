"""
Houdini AI VEX Generator Shelf Tool
Version: 1.1
Date: 2024-06-12
Description: Generates VEX code using selected AI provider (Anthropic, Gemini, or OpenAI).
             Applies generated code to a new or selected Attribute Wrangle node.

Requirements:
- Houdini 18.0+ (for PySide2)
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
"""

import hou
import os
import sys
import traceback
import platform

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui # Fallback for older Houdini/PyQt installs
    except ImportError:
        print("ERROR: Could not import PySide2 or PyQt5. This tool requires a Houdini version with a working Qt binding.")
        # Optionally raise an error or exit if hou is available but Qt isn't
        if hou.isUIAvailable():
             hou.ui.displayMessage("Fatal Error: PySide2/PyQt5 not found.\nCannot run AI VEX Generator.",
                                   title="Missing UI Library", severity=hou.severityType.Error)
        # Cannot proceed without Qt
        # Consider raising an exception here if this script is imported elsewhere
        # raise ImportError("Missing PySide2/PyQt5")


# --- API Key Configuration (Check Environment Variables First) ---
MANUAL_ANTHROPIC_API_KEY = "YOUR API KEY"  # Your Anthropic API key
MANUAL_GOOGLE_API_KEY = "YOUR API KEY"     # Your Google AI API key
MANUAL_OPENAI_API_KEY = "YOUR API KEY"     # Your OpenAI API key

# --- Model Configuration ---
# You can change these to your preferred models or newer versions
ANTHROPIC_MODEL = "claude-3-haiku-20240307" # Faster and cheaper Opus alternative
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"       # Or "gemini-1.5-flash-latest" for speed
OPENAI_MODEL = "gpt-4o"                      # Current flagship model

# --- Library Import with Error Handling ---
anthropic = None
genai = None
openai = None
missing_libs = []

try:
    import anthropic
    print("INFO: Successfully imported anthropic library.")
except ImportError:
    print("WARNING: anthropic library not found. Install via Houdini's Python: hython -m pip install anthropic")
    missing_libs.append("anthropic")

try:
    import google.generativeai as genai
    print("INFO: Successfully imported google.generativeai library.")
except ImportError:
    print("WARNING: google-generativeai library not found. Install via Houdini's Python: hython -m pip install google-generativeai")
    missing_libs.append("google-generativeai")

try:
    import openai
    # Check OpenAI library version (needs >= 1.0.0)
    try:
        if openai.__version__ < "1.0.0":
            print(f"WARNING: OpenAI library version {openai.__version__} is outdated. This script requires >= 1.0.0. Please upgrade: hython -m pip install --upgrade openai")
            # Keep openai as None to prevent using the old version's incompatible API
            openai = None
            missing_libs.append("openai (Update Required)")
        else:
            print(f"INFO: Successfully imported openai library (version {openai.__version__}).")
    except AttributeError:
        # Handle cases where __version__ might not be present (though unlikely for openai)
        print("INFO: Successfully imported openai library (version could not be determined). Assuming >= 1.0.0.")

except ImportError:
    print("WARNING: openai library not found. Install via Houdini's Python: hython -m pip install openai")
    missing_libs.append("openai")


# --- Core Functions ---

def get_api_key(provider_name):
    """Get API key from environment or manual setting."""
    api_key = None
    key_source = "Not found"

    if provider_name == "Anthropic Claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            key_source = "Environment Variable (ANTHROPIC_API_KEY)"
        elif MANUAL_ANTHROPIC_API_KEY:
            api_key = MANUAL_ANTHROPIC_API_KEY
            key_source = "Manual Script Setting (Less Secure)"
    elif provider_name == "Google Gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            key_source = "Environment Variable (GOOGLE_API_KEY)"
        elif MANUAL_GOOGLE_API_KEY:
            api_key = MANUAL_GOOGLE_API_KEY
            key_source = "Manual Script Setting (Less Secure)"
    elif provider_name == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            key_source = "Environment Variable (OPENAI_API_KEY)"
        elif MANUAL_OPENAI_API_KEY:
            api_key = MANUAL_OPENAI_API_KEY
            key_source = "Manual Script Setting (Less Secure)"

    if api_key:
        print(f"INFO: Using API key for {provider_name} from: {key_source}")
        if "Manual" in key_source:
            print("WARNING: Using a hardcoded API key is less secure than using environment variables.")
    else:
        print(f"ERROR: No API key found for {provider_name}. Set the corresponding environment variable or the manual fallback in the script.")

    return api_key

def clean_vex_response(text):
    """Clean up AI response to extract only the VEX code."""
    if not text:
        return "" # Return empty string instead of None

    text = text.strip()

    # Handle markdown code fences (```vex, ```c, ```cpp, ```)
    lines = text.splitlines()
    if not lines:
        return ""

    start_index = 0
    end_index = len(lines)

    # Detect start fence
    first_line_lower = lines[0].lower().strip()
    if first_line_lower.startswith("```"):
        # Check for language specifier (vex, c, cpp) which might be attached or on its own
        potential_lang = first_line_lower[3:].strip()
        if potential_lang in ["vex", "c", "cpp", "c++", ""]:
            start_index = 1 # Skip the opening fence line

    # Detect end fence
    # Iterate backwards to find the last potential fence
    for i in range(len(lines) - 1, start_index -1, -1):
         if lines[i].strip() == "```":
             end_index = i # Exclude the closing fence line
             break # Found the last fence

    # Handle case where only ``` fences were present but no actual code
    if start_index >= end_index and (lines[0].strip().startswith("```") or lines[-1].strip() == "```") :
         return "" # Likely just fences, return empty

    cleaned_lines = lines[start_index:end_index]
    return "\n".join(cleaned_lines).strip() # Join and strip again


def call_anthropic_api(api_key, prompt):
    """Generate VEX code using Anthropic Claude."""
    if not anthropic:
        raise ImportError("Anthropic library is not installed or failed to import.")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are an expert Houdini VEX programmer. Your sole task is to generate VEX code snippet suitable for a Houdini Attribute Wrangle node.
Respond ONLY with the raw VEX code.
Do NOT include any explanations, introductions, markdown formatting (like ```vex or ```), or any text other than the VEX code itself.
The code must be complete and syntactically correct VEX.
Use standard VEX functions available in Houdini.
Include minimal, concise comments within the VEX code using // or /* */ for clarity where necessary."""

    try:
        print(f"INFO: Sending prompt to Anthropic Claude (Model: {ANTHROPIC_MODEL})")
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096, # Increased slightly
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Generate VEX code based on this description: {prompt}"}]
        )

        reply_text = ""
        if response.content:
             for content_block in response.content:
                 if content_block.type == "text":
                     reply_text += content_block.text
        else:
             print("WARNING: Anthropic API returned no content blocks.")
             return "" # Return empty string if no content

        cleaned_code = clean_vex_response(reply_text)
        if not cleaned_code:
             print("WARNING: Anthropic response was empty after cleaning.")
             # Optionally, return the raw response if cleaning removes everything unexpectedly
             # return reply_text # Uncomment this line to debug cleaning issues
        return cleaned_code


    except anthropic.APIConnectionError as e:
        print(f"ERROR: Anthropic API connection failed: {e}")
        raise ConnectionError(f"Anthropic connection error: {e}") from e
    except anthropic.RateLimitError as e:
        print(f"ERROR: Anthropic rate limit exceeded: {e}")
        raise ConnectionAbortedError(f"Anthropic rate limit error: {e}") from e # Or a custom exception
    except anthropic.APIStatusError as e:
        print(f"ERROR: Anthropic API error (Status {e.status_code}): {e.message}")
        raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}") from e
    except Exception as e:
        print(f"ERROR: Unexpected error during Anthropic API call: {str(e)}")
        traceback.print_exc()
        raise


def call_gemini_api(api_key, prompt):
    """Generate VEX code using Google Gemini."""
    if not genai:
        raise ImportError("Google Generative AI library is not installed or failed to import.")

    try:
        genai.configure(api_key=api_key)

        system_instruction = """You are an expert Houdini VEX programmer. Your sole task is to generate a VEX code snippet suitable for a Houdini Attribute Wrangle node.
Respond ONLY with the raw VEX code.
Do NOT include any explanations, introductions, markdown formatting (like ```vex or ```), or any text other than the VEX code itself.
The code must be complete and syntactically correct VEX.
Use standard VEX functions available in Houdini.
Include minimal, concise comments within the VEX code using // or /* */ for clarity where necessary."""

        # Configure safety settings to be less restrictive for code generation
        safety_settings = {
            # Threshold options: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH
            # Using BLOCK_ONLY_HIGH for potentially sensitive categories
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_ONLY_HIGH',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_ONLY_HIGH',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH',
             # Be cautious with DANGEROUS_CONTENT if generating code that might involve system interaction (less relevant for VEX)
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_ONLY_HIGH',
        }

        # Initialize the model with system instruction
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system_instruction,
            safety_settings=safety_settings
        )

        generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192 # Gemini 1.5 Pro has a large context, allow more output
        )


        print(f"INFO: Sending prompt to Google Gemini (Model: {GEMINI_MODEL})")
        # Send only the user prompt, system instruction is handled by the model instance
        response = model.generate_content(
             f"Generate VEX code based on this description: {prompt}",
             generation_config=generation_config
         )

        reply_text = ""
        # Accessing response text robustly
        try:
            if hasattr(response, 'text'):
                 reply_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                 # Concatenate text from parts if available
                 reply_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                 print(f"WARNING: Gemini response structure unexpected or empty. Response: {response}")
                 # Check for blocked prompts
                 if response.prompt_feedback.block_reason:
                      block_msg = f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}"
                      print(f"ERROR: {block_msg}")
                      # Check safety ratings for details
                      for rating in response.prompt_feedback.safety_ratings:
                           print(f" - {rating.category}: {rating.probability}")
                      raise ValueError(block_msg) # Raise specific error for blocked content
                 else:
                      return "" # Return empty if no text and not blocked

        except ValueError as ve:
            # Handle potential errors during text extraction or if response indicates blocking
            print(f"ERROR: Error processing Gemini response: {ve}")
            if "block_reason" in str(ve): # Re-raise the blocking error
                raise ve
            return "" # Return empty for other value errors during processing

        cleaned_code = clean_vex_response(reply_text)
        if not cleaned_code and reply_text: # Log if cleaning removed everything
             print(f"WARNING: Gemini response was empty after cleaning. Original response was:\n---\n{reply_text}\n---")
        elif not cleaned_code:
             print("WARNING: Gemini response was empty after cleaning (and likely before).")

        return cleaned_code

    except ImportError as e:
         # This case handles if genai was initially imported but configuration fails somehow
         print(f"ERROR: Failed to configure or use Google GenAI library: {e}")
         raise
    except Exception as e:
        # Catch other potential exceptions from the API call or configuration
        print(f"ERROR: Google Gemini API call failed: {str(e)}")
        # Look for specific Google API errors if the library provides them
        # Example: isinstance(e, google.api_core.exceptions.GoogleAPICallError)
        traceback.print_exc()
        raise # Re-raise the exception to be caught by the main UI handler


def call_openai_api(api_key, prompt):
    """Generate VEX code using OpenAI (v1.0+ library)."""
    if not openai:
        raise ImportError("OpenAI library is not installed, failed to import, or is outdated (< 1.0.0).")

    try:
        # Instantiate the client with the API key (v1.x syntax)
        client = openai.OpenAI(api_key=api_key)
    except Exception as e_client:
        print(f"ERROR: Failed to initialize OpenAI client: {str(e_client)}")
        # Provide more specific guidance if it's an authentication error
        if "Incorrect API key" in str(e_client):
             raise ValueError(f"OpenAI Authentication Error: Incorrect API key provided. Please check your key.") from e_client
        else:
             raise ImportError(f"Could not initialize OpenAI client: {e_client}") from e_client


    system_prompt = """You are an expert Houdini VEX programmer. Your sole task is to generate a VEX code snippet suitable for a Houdini Attribute Wrangle node.
Respond ONLY with the raw VEX code.
Do NOT include any explanations, introductions, markdown formatting (like ```vex or ```), or any text other than the VEX code itself.
The code must be complete and syntactically correct VEX.
Use standard VEX functions available in Houdini.
Include minimal, concise comments within the VEX code using // or /* */ for clarity where necessary."""

    try:
        print(f"INFO: Sending prompt to OpenAI (Model: {OPENAI_MODEL})")
        # Use the client's chat.completions.create method (v1.x syntax)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=4096, # Response max_tokens
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate VEX code based on this description: {prompt}"}
            ]
        )

        # Access the response content using the v1.x structure
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            reply_text = response.choices[0].message.content
            cleaned_code = clean_vex_response(reply_text)
            if not cleaned_code and reply_text:
                 print(f"WARNING: OpenAI response was empty after cleaning. Original response was:\n---\n{reply_text}\n---")
            elif not cleaned_code:
                 print("WARNING: OpenAI response was empty after cleaning (and likely before).")
            return cleaned_code
        else:
             # Log finish reason if available
             finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
             print(f"WARNING: Received empty or invalid response structure from OpenAI. Finish reason: {finish_reason}")
             # Example: If finish_reason is 'content_filter', it was blocked.
             if finish_reason == 'content_filter':
                  raise ValueError("OpenAI request blocked due to content filter.")
             # Example: If finish_reason is 'length', max_tokens might be too small.
             elif finish_reason == 'length':
                  print("WARNING: OpenAI response may be truncated due to token limits.")
                  # Try to return potentially partial content if available (handled above)
                  # If reply_text was extracted but cleaned to empty, this case won't be reached
             return "" # Return empty for other unexpected structures


    # Specific OpenAI v1.x Error Handling
    except openai.APIStatusError as e:
         message = f"OpenAI API Error (Status {e.status_code}): {e.message}"
         print(f"ERROR: {message}")
         # Append details from response if available
         if hasattr(e, 'response') and e.response:
             print(f"Response Body: {e.response.text}") # Be careful logging full response bodies
         raise ConnectionAbortedError(message) from e # Re-raise a more general type if needed
    except openai.APIConnectionError as e:
        message = f"Failed to connect to OpenAI API: {e}"
        print(f"ERROR: {message}")
        raise ConnectionError(message) from e
    except openai.RateLimitError as e:
        message = f"OpenAI Rate Limit Exceeded: {e}"
        print(f"ERROR: {message}")
        raise ConnectionAbortedError(message) from e # Or custom exception
    except openai.AuthenticationError as e:
         message = f"OpenAI Authentication Error: {e}. Check your API key."
         print(f"ERROR: {message}")
         raise ValueError(message) from e # Use ValueError for auth issues
    except ValueError as ve: # Catch specific value errors raised above (e.g., content filter)
         print(f"ERROR: {str(ve)}")
         raise # Re-raise the specific ValueError
    except Exception as e: # Catch any other unexpected errors
        print(f"ERROR: Unexpected error during OpenAI API call: {str(e)}")
        traceback.print_exc()
        raise # Re-raise the original exception


def apply_vex_to_new_wrangle(vex_code):
    """Creates a new Attribute Wrangle node and applies the VEX code."""
    selected_nodes = hou.selectedNodes()

    if not selected_nodes:
        hou.ui.displayMessage("Please select the node to connect the new Wrangle node after.",
                             title="Input Node Required", severity=hou.severityType.Warning)
        return None # Return None to indicate failure

    if len(selected_nodes) > 1:
        hou.ui.displayMessage("Please select only one node to connect the new Wrangle after.",
                             title="Multiple Nodes Selected", severity=hou.severityType.Warning)
        return None

    input_node = selected_nodes[0]
    parent = input_node.parent()

    try:
        # Create new Attribute Wrangle node
        wrangle_node = parent.createNode("attribwrangle", "ai_generated_wrangle")

        # Position it below the selected node
        input_pos = input_node.position()
        wrangle_node.setPosition([input_pos[0], input_pos[1] - 1.0]) # Standard node spacing

        # Connect input node to the first input of the wrangle
        wrangle_node.setInput(0, input_node)

        # Set the VEX code snippet
        snippet_parm = wrangle_node.parm("snippet")
        if snippet_parm:
            snippet_parm.set(vex_code)
        else:
            # Should not happen for attribwrangle, but good practice
            raise hou.ParmNotFound("Could not find 'snippet' parameter on the created node.")

        # Set display and render flags for the new node
        wrangle_node.setDisplayFlag(True)
        wrangle_node.setRenderFlag(True)

        # Select the newly created node
        wrangle_node.setSelected(True, clear_all_selected=True)

        # Move the network view to the new node
        network_editor = hou.ui.paneTabOfType(hou.paneTabType.NetworkEditor)
        if network_editor:
            network_editor.setCurrentNode(wrangle_node)
            # Frame selection can sometimes be useful too
            # network_editor.frameSelection()

        print(f"INFO: Created wrangle node '{wrangle_node.name()}' with generated VEX.")
        return wrangle_node # Return the created node on success

    except hou.OperationFailed as e:
        # Specific Houdini operation errors
        err_msg = f"Error creating or configuring Wrangle node: {str(e)}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        hou.ui.displayMessage(err_msg, title="Node Creation Error", severity=hou.severityType.Error)
        return None
    except Exception as e:
        # General errors
        err_msg = f"An unexpected error occurred during node creation: {str(e)}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        hou.ui.displayMessage(err_msg, title="Node Creation Error", severity=hou.severityType.Error)
        return None


def apply_to_selected_wrangle(vex_code):
    """Applies VEX code to the currently selected Attribute Wrangle node."""
    selected_nodes = hou.selectedNodes()

    if not selected_nodes:
        hou.ui.displayMessage("Please select an Attribute Wrangle node first.",
                             title="Selection Required", severity=hou.severityType.Warning)
        return False

    if len(selected_nodes) > 1:
        hou.ui.displayMessage("Please select only one node.",
                             title="Multiple Nodes Selected", severity=hou.severityType.Warning)
        return False

    node = selected_nodes[0]

    # Check if the selected node is likely a wrangle (has a 'snippet' parm)
    snippet_parm = node.parm("snippet")
    if not snippet_parm:
        hou.ui.displayMessage(f"Selected node '{node.name()}' of type '{node.type().name()}' "
                             f"does not have a 'snippet' parameter.\nPlease select an Attribute Wrangle node.",
                             title="Incompatible Node", severity=hou.severityType.Warning)
        return False

    try:
        # Set the VEX code
        snippet_parm.set(vex_code)
        print(f"INFO: Applied generated VEX code to selected node '{node.name()}'.")

        # Optional: Briefly flash the node to indicate success
        # node.flashColor(hou.Color(0.2, 0.8, 0.2)) # Green flash

        return True

    except hou.PermissionError as e:
        err_msg = f"Error applying code to '{node.name()}': Parameter is locked or node is inside a locked HDA.\n{str(e)}"
        print(f"ERROR: {err_msg}")
        hou.ui.displayMessage(err_msg, title="Permission Error", severity=hou.severityType.Error)
        return False
    except Exception as e:
        err_msg = f"An unexpected error occurred while applying code: {str(e)}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        hou.ui.displayMessage(err_msg, title="Parameter Error", severity=hou.severityType.Error)
        return False


# --- UI Class ---

class AIVEXGeneratorDialog(QtWidgets.QDialog):
    def __init__(self, available_providers, parent=None):
        super(AIVEXGeneratorDialog, self).__init__(parent)

        self.providers = available_providers
        self.last_used_provider = None # Store the last used provider

        self.setWindowTitle("AI VEX Code Generator v1.1")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowMinimizeButtonHint) # Add Max/Min buttons
        self.setMinimumSize(750, 550)
        # Make the dialog modal (blocks interaction with Houdini main window)
        self.setModal(True)

        # Check if running on macOS for potential font adjustments
        self.is_macos = platform.system() == "Darwin"

        self.setup_ui()
        self.load_settings() # Load last used provider


    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10) # Add some margins
        main_layout.setSpacing(8) # Adjust spacing between elements

        # --- Provider Selection ---
        provider_layout = QtWidgets.QHBoxLayout()
        provider_label = QtWidgets.QLabel("AI Provider:")
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems(self.providers)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo, 1) # Give combo box stretch factor
        # provider_layout.addStretch() # No longer needed with stretch factor

        main_layout.addLayout(provider_layout)

        # --- Prompt Area ---
        prompt_label = QtWidgets.QLabel("Describe the VEX functionality:")
        main_layout.addWidget(prompt_label)

        self.prompt_edit = QtWidgets.QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Example: 'Color points based on their height (Y position). Red at the bottom, green at the top.' "
            "or 'Create velocity vectors pointing away from the center of the bounding box.'"
        )
        self.prompt_edit.setMinimumHeight(80)
        self.prompt_edit.setMaximumHeight(200) # Limit prompt height
        main_layout.addWidget(self.prompt_edit)

        # --- Generate Button ---
        self.generate_button = QtWidgets.QPushButton("Generate VEX Code")
        self.generate_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 15px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #3e8e41; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
            )
        self.generate_button.clicked.connect(self.on_generate_clicked)
        main_layout.addWidget(self.generate_button)

        # --- Response Area ---
        response_label = QtWidgets.QLabel("Generated VEX Code (Editable):")
        main_layout.addWidget(response_label)

        self.response_edit = QtWidgets.QTextEdit()
        self.response_edit.setReadOnly(False) # Keep editable for quick fixes
        self.response_edit.setAcceptRichText(False) # Ensure plain text
        self.response_edit.setMinimumHeight(200)

        # Set monospace font
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        # Adjust point size slightly - might need tweaking per OS
        font.setPointSize(10 if not self.is_macos else 12) # Slightly larger on macOS often looks better
        self.response_edit.setFont(font)

        main_layout.addWidget(self.response_edit, 1) # Give response area stretch factor

        # --- Status Label ---
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setStyleSheet("color: #333333; padding: 3px;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # --- Action Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)

        self.copy_button = QtWidgets.QPushButton("Copy Code")
        self.copy_button.setToolTip("Copy the generated VEX code to the clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)

        self.new_wrangle_button = QtWidgets.QPushButton("Create New Wrangle")
        self.new_wrangle_button.setToolTip("Create a new Attribute Wrangle node after the selected node and apply the code")
        self.new_wrangle_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px 15px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1e88e5; }"
            "QPushButton:pressed { background-color: #1976d2; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
            )
        self.new_wrangle_button.clicked.connect(self.create_new_wrangle)

        self.apply_button = QtWidgets.QPushButton("Apply to Selected")
        self.apply_button.setToolTip("Apply the generated VEX code to the selected Attribute Wrangle node")
        self.apply_button.setStyleSheet(
             "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px 15px; border-radius: 3px; }"
             "QPushButton:hover { background-color: #fb8c00; }"
             "QPushButton:pressed { background-color: #f57c00; }"
             "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
             )
        self.apply_button.clicked.connect(self.apply_to_selected)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.reject) # reject() closes the dialog

        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.new_wrangle_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch() # Pushes close button to the right
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

        # --- Initial State ---
        self.enable_output_actions(False) # Disable actions until code is generated


    def enable_controls(self, enabled):
        """Enable/disable input controls during generation."""
        self.provider_combo.setEnabled(enabled)
        self.prompt_edit.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        # Also enable/disable output actions based on whether there's text
        has_text = bool(self.response_edit.toPlainText().strip())
        self.enable_output_actions(enabled and has_text)


    def enable_output_actions(self, enabled):
        """Enable or disable output-related actions."""
        self.copy_button.setEnabled(enabled)
        self.new_wrangle_button.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)


    def update_status(self, message, level="info"):
        """Update the status label with styled messages."""
        if level == "info":
            self.status_label.setStyleSheet("color: #333333; padding: 3px;") # Dark gray/black
        elif level == "success":
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 3px;") # Green
        elif level == "warning":
             self.status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 3px;") # Orange
        elif level == "error":
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold; padding: 3px;") # Red
        else: # Default back to info
             self.status_label.setStyleSheet("color: #333333; padding: 3px;")

        self.status_label.setText(message)
        QtWidgets.QApplication.processEvents() # Force UI update


    def on_generate_clicked(self):
        """Handle the 'Generate VEX Code' button click."""
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            hou.ui.displayMessage("Please enter a description for the VEX code you need.",
                                 title="Empty Prompt", severity=hou.severityType.Warning)
            self.update_status("Prompt cannot be empty.", level="warning")
            return

        provider = self.provider_combo.currentText()
        self.last_used_provider = provider # Store for saving settings
        api_key = get_api_key(provider)

        if not api_key:
            err_msg = (f"No API key found for {provider}.\n\n"
                       f"Please configure the API key using environment variables (recommended) "
                       f"or directly in the script (less secure). See script comments for details.")
            hou.ui.displayMessage(err_msg, title="API Key Missing", severity=hou.severityType.Error)
            self.update_status(f"API key missing for {provider}.", level="error")
            return

        # --- Start Generation Process ---
        self.enable_controls(False)
        self.generate_button.setText("Generating...")
        self.response_edit.setPlainText("") # Clear previous response
        self.update_status(f"Sending request to {provider}...", level="info")

        vex_code = None
        error_occurred = False
        error_message = "An unknown error occurred."

        try:
            # Call the appropriate API function
            if provider == "Anthropic Claude":
                vex_code = call_anthropic_api(api_key, prompt)
            elif provider == "Google Gemini":
                vex_code = call_gemini_api(api_key, prompt)
            elif provider == "OpenAI":
                vex_code = call_openai_api(api_key, prompt)
            else:
                # This case should ideally not be reachable if combo box is populated correctly
                raise ValueError(f"Unknown provider selected: {provider}")

            # Check if code generation was successful (returned non-empty string)
            if vex_code is not None and vex_code.strip(): # Check if not None and not empty after strip
                 self.response_edit.setPlainText(vex_code)
                 self.update_status("VEX code generated successfully!", level="success")
                 self.enable_output_actions(True) # Enable actions now that there's code
            elif vex_code is None: # Explicit None might indicate an API issue before returning text
                 error_message = "API call failed or returned an unexpected result. Check console."
                 print(f"ERROR: API call for {provider} returned None.")
                 self.update_status(error_message, level="error")
                 error_occurred = True
            else: # Empty string returned, potentially valid (empty code) or failed cleaning
                 self.response_edit.setPlainText("") # Ensure it's empty
                 warning_message = "Generated code is empty. The AI might not have understood the request or the response was filtered/cleaned."
                 print(f"WARNING: {warning_message} (Provider: {provider})")
                 self.update_status(warning_message, level="warning")
                 # Keep output actions disabled for empty string result
                 self.enable_output_actions(False)


        except ImportError as e:
            error_message = f"Library Error: {str(e)}"
            print(f"ERROR: {error_message}")
            traceback.print_exc()
            hou.ui.displayMessage(f"{error_message}\nPlease ensure the required library is installed correctly.", title="Import Error", severity=hou.severityType.Error)
            self.update_status(error_message, level="error")
            error_occurred = True
        except (ConnectionError, ConnectionAbortedError, ValueError, RuntimeError) as e:
             # Catch specific errors raised by the API functions
             error_message = f"API Error ({provider}): {str(e)}"
             print(f"ERROR: {error_message}")
             # Don't need traceback here usually as it's printed in the API function
             hou.ui.displayMessage(error_message, title="API Communication Error", severity=hou.severityType.Error)
             self.update_status(f"API Error: {str(e)}", level="error")
             error_occurred = True
        except Exception as e:
            # Catch any other unexpected exceptions
            error_message = f"Unexpected Error: {str(e)}"
            print(f"ERROR: {error_message}")
            traceback.print_exc() # Print detailed traceback for unexpected errors
            hou.ui.displayMessage(f"{error_message}\nSee Houdini console for more details.", title="Generation Error", severity=hou.severityType.Error)
            self.update_status(f"Error: {str(e)}", level="error")
            error_occurred = True

        finally:
            # --- Finish Generation Process ---
            # Re-enable controls, except output actions if an error occurred or code is empty
            self.enable_controls(True)
            self.generate_button.setText("Generate VEX Code")
            # Ensure output actions are disabled if there was an error or no code
            if error_occurred or not self.response_edit.toPlainText().strip():
                 self.enable_output_actions(False)


    def copy_to_clipboard(self):
        """Copy the content of the response text edit to the system clipboard."""
        vex_code = self.response_edit.toPlainText()
        if not vex_code:
             self.update_status("Nothing to copy.", level="warning")
             return

        try:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(vex_code)
            self.update_status("Code copied to clipboard!", level="success")
        except Exception as e:
             error_msg = f"Failed to copy to clipboard: {e}"
             print(f"ERROR: {error_msg}")
             self.update_status(error_msg, level="error")


    def create_new_wrangle(self):
        """Create a new wrangle node with the generated VEX code."""
        vex_code = self.response_edit.toPlainText()
        if not vex_code.strip():
            self.update_status("Cannot create wrangle: Generated code is empty.", level="warning")
            return

        created_node = apply_vex_to_new_wrangle(vex_code)
        if created_node:
            self.update_status(f"Created new wrangle '{created_node.name()}'.", level="success")
            self.save_settings() # Save settings on successful action
            self.accept() # Close the dialog on success
        else:
            # Error message is displayed by apply_vex_to_new_wrangle
            self.update_status("Failed to create new wrangle node. See console/popup.", level="error")


    def apply_to_selected(self):
        """Apply the generated VEX code to the selected wrangle node."""
        vex_code = self.response_edit.toPlainText()
        if not vex_code.strip():
            self.update_status("Cannot apply: Generated code is empty.", level="warning")
            return

        if apply_to_selected_wrangle(vex_code):
            self.update_status("Applied VEX code to selected node.", level="success")
            self.save_settings() # Save settings on successful action
            self.accept() # Close the dialog on success
        else:
            # Error message is displayed by apply_to_selected_wrangle
            self.update_status("Failed to apply code to selected node. See console/popup.", level="error")

    # --- Settings Persistence (Simple Example) ---
    def save_settings(self):
        """Save minimal settings (like last provider) to Houdini user prefs."""
        if self.last_used_provider:
             try:
                 # Use a unique key for this tool's settings
                 hou.hscript(f"hsc Setaivxgenlastprovider = '{self.last_used_provider}'")
                 print(f"INFO: Saved last used provider: {self.last_used_provider}")
             except hou.Error as e:
                 print(f"WARNING: Could not save settings: {e}")

    def load_settings(self):
        """Load minimal settings from Houdini user prefs."""
        try:
            # Note: hscript returns a tuple (output, error)
            result, _ = hou.hscript("hsc Echo $aivxgenlastprovider")
            last_provider = result.strip()
            if last_provider in self.providers:
                 index = self.provider_combo.findText(last_provider)
                 if index != -1:
                     self.provider_combo.setCurrentIndex(index)
                     self.last_used_provider = last_provider
                     print(f"INFO: Loaded last used provider: {last_provider}")
            else:
                 print("INFO: No valid saved provider found or provider no longer available.")

        except hou.Error as e:
            # This is expected if the variable hasn't been set yet
            print(f"INFO: Could not load settings (likely first run): {e}")
        except Exception as e:
             print(f"WARNING: Unexpected error loading settings: {e}")


    # Override closeEvent to ensure settings are saved when dialog is closed via 'X'
    def closeEvent(self, event):
        self.save_settings()
        super(AIVEXGeneratorDialog, self).closeEvent(event)

    # Override reject to ensure settings are saved when 'Close' button or ESC is pressed
    def reject(self):
        self.save_settings()
        super(AIVEXGeneratorDialog, self).reject()


# --- Main Entry Point ---

# Global variable to store the dialog instance to prevent multiple instances
_ai_vex_dialog_instance = None

def run_ai_vex_tool():
    """Main function to launch the AI VEX Generator dialog."""
    global _ai_vex_dialog_instance

    if not hou.isUIAvailable():
        print("ERROR: This tool requires a Houdini UI session (Houdini FX, Core, etc.). Cannot run in hython or non-UI environments.")
        return

    # Determine available providers based on successful imports
    available_providers = []
    if anthropic:
        available_providers.append("Anthropic Claude")
    if genai:
        available_providers.append("Google Gemini")
    if openai:
        # Check again if openai object exists (might be None due to version check)
        if openai:
            available_providers.append("OpenAI")
        else:
             # Add a note about the version issue if it was the cause
             if "openai (Update Required)" in missing_libs:
                  print("INFO: OpenAI provider disabled due to outdated library version.")
             # else: # Should have been caught by import error
             #      print("INFO: OpenAI provider disabled because the library failed to import.")


    if not available_providers:
        install_instructions = (
            "ERROR: No AI provider libraries found or usable.\n\n"
            "Please install at least one of the following Python packages using Houdini's Python environment:\n\n"
            "1. Open Houdini's Command Line Tools (or a terminal where 'hython' is in PATH).\n"
            "2. Run the install command(s):\n"
            "   For Anthropic Claude:\n     hython -m pip install anthropic\n\n"
            "   For Google Gemini:\n     hython -m pip install google-generativeai\n\n"
            "   For OpenAI (ensure version >= 1.0.0):\n     hython -m pip install --upgrade openai\n\n"
            "3. Restart Houdini after installation.\n\n"
            f"Missing/Problematic Libraries: {', '.join(missing_libs)}"
        )
        print(install_instructions)
        hou.ui.displayMessage(install_instructions, title="Missing Dependencies", severity=hou.severityType.Error, details_label="Installation Steps", details=install_instructions)
        return

    # Check if an instance already exists and bring it to front
    # Uses weakref if available, otherwise stores instance directly (careful with GC)
    if _ai_vex_dialog_instance:
        try:
             # Check if the window still exists (wasn't closed)
             if _ai_vex_dialog_instance.isVisible():
                 print("INFO: AI VEX Generator dialog already open. Bringing to front.")
                 _ai_vex_dialog_instance.show()
                 _ai_vex_dialog_instance.raise_() # Bring window to the front
                 _ai_vex_dialog_instance.activateWindow() # Activate it
                 return
             else:
                 # The instance exists but the window was closed, allow creating a new one
                 print("INFO: Stale dialog instance found. Creating a new one.")
                 _ai_vex_dialog_instance = None # Clear the old reference
        except RuntimeError: # Catches 'Internal C++ object already deleted'
             print("INFO: Stale dialog instance (C++ object deleted). Creating a new one.")
             _ai_vex_dialog_instance = None # Clear the invalid reference


    # Create and show the dialog
    try:
        # Pass the list of available providers to the dialog
        dialog = AIVEXGeneratorDialog(available_providers, parent=hou.qt.mainWindow())
        _ai_vex_dialog_instance = dialog # Store the new instance
        # Use exec_() for modal dialog behavior (blocks until closed)
        dialog.exec_()
        # Dialog is closed here, clear the instance reference if using exec_()
        # If using show() instead, clearing might happen elsewhere or rely on Python's GC
        _ai_vex_dialog_instance = None


    except Exception as e:
        error_msg = f"FATAL ERROR: Could not launch AI VEX Generator dialog.\n\nError: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        if hou.isUIAvailable():
            hou.ui.displayMessage(error_msg, title="Script Execution Error", severity=hou.severityType.Error, details=traceback.format_exc())
        # Ensure instance is cleared on error
        _ai_vex_dialog_instance = None


# --- Allow running directly for testing if needed ---
# Note: This won't work correctly outside a Houdini UI environment
# if __name__ == "__main__":
#     # This block is mainly for IDE/static analysis; direct execution
#     # is unlikely to work without a running Houdini instance.
#     print("Script intended to be run from within Houdini.")
#     # Example pseudo-call for testing structure (requires mocking hou)
#     # if hou.isUIAvailable(): run_ai_vex_tool()