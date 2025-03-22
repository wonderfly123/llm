from flask import Flask, request, jsonify
import os
import time
import uuid
import google.generativeai as genai

app = Flask(__name__)

class GeminiFlashNameProcessor:
    def __init__(self, api_key):
        """
        Initialize Gemini 2.0 Flash for name processing
        """
        genai.configure(api_key=api_key)
        
        # Specifically use Gemini 2.0 Flash model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def process_name(self, raw_name):
        """
        Process name using Gemini 2.0 Flash
        
        Args:
            raw_name (str): Raw name input
        
        Returns:
            str: Processed name
        """
        # Detailed prompt for name processing
        prompt = f"""
        CRITICAL INSTRUCTION: You shall save ALL text EXACTLY as it is spelled.

        Absolute Preservation Rules:
        - NEVER modify ANY spelling
        - Preserve EXACTLY how text was input
        - This applies to:
          1. Personal Names
          2. Street Names
          3. City Names
          4. Any other text input

        Specific Requirements:
        - Do NOT change capitalization
        - Do NOT remove or add spaces
        - Do NOT alter any characters
        - Return text PRECISELY as it was originally spelled

        Input Text: {raw_name}
        
        Processed Text:
        """
        
        try:
            # Generate response using Gemini 2.0 Flash
            response = self.model.generate_content(prompt)
            processed_name = response.text.strip()
            
            return processed_name
        
        except Exception as e:
            # Fallback processing
            print(f"Error processing name: {e}")
            return raw_name

# Initialize processor with API key
name_processor = GeminiFlashNameProcessor(os.environ.get('GOOGLE_AI_API_KEY'))

@app.route('/', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({"status": "healthy"}), 200

@app.route('/v1/chat/completions', methods=['POST'])
def process_name_endpoint():
    """
    Endpoint to match OpenAI chat completions API format
    """
    try:
        # Extract request data
        data = request.json
        
        # Extract the last message content
        messages = data.get('messages', [])
        raw_name = messages[-1]['content'] if messages else ''
        
        # Process the name
        processed_name = name_processor.process_name(raw_name)
        
        # Construct response in OpenAI API format
        response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemini-2.0-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": processed_name,
                    "refusal": None,
                    "annotations": []
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(raw_name.split()),
                "completion_tokens": len(processed_name.split()),
                "total_tokens": len(raw_name.split()) + len(processed_name.split()),
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "service_tier": "default"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
