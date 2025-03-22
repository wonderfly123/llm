from flask import Flask, request, jsonify
import os
import time
import uuid
import logging
import traceback
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class GeminiFlashNameProcessor:
    def __init__(self, api_key):
        """
        Initialize Gemini 2.0 Flash for name processing
        """
        try:
            genai.configure(api_key=api_key)
            
            # Specifically use Gemini 2.0 Flash model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_name(self, raw_name):
        """
        Process name using Gemini 2.0 Flash
        
        Args:
            raw_name (str): Raw name input
        
        Returns:
            str: Processed name
        """
        logger.info(f"Processing input: {raw_name}")
        
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
            
            logger.info(f"Processed output: {processed_name}")
            return processed_name
        
        except Exception as e:
            logger.error(f"Error processing name: {e}")
            logger.error(traceback.format_exc())
            return raw_name

# Initialize processor with API key
try:
    name_processor = GeminiFlashNameProcessor(os.environ.get('GOOGLE_AI_API_KEY'))
except Exception as e:
    logger.error(f"Failed to initialize name processor: {e}")
    logger.error(traceback.format_exc())
    name_processor = None

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
        # Log full request details
        logger.info("Received request to /v1/chat/completions")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Extract request data
        try:
            data = request.get_json()
            logger.info(f"Request JSON data: {data}")
        except Exception as parse_error:
            logger.error(f"Error parsing JSON: {parse_error}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Check if name processor is initialized
        if name_processor is None:
            logger.error("Name processor not initialized")
            return jsonify({
                "error": "Name processor initialization failed"
            }), 500
        
        # Extract the last message content
        messages = data.get('messages', [])
        logger.info(f"Messages: {messages}")
        
        raw_name = messages[-1]['content'] if messages else ''
        logger.info(f"Raw name: {raw_name}")
        
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
        
        logger.info(f"Final response: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unhandled error processing request: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
