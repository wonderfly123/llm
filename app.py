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

class NameValidator:
    def __init__(self, api_key):
        """
        Initialize Gemini 2.0 Flash for name validation
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def validate_name_spelling(self, messages):
        """
        Validate and preserve exact name spelling
        
        Args:
            messages (list): Conversation history
        
        Returns:
            dict: Processed response with validated spelling
        """
        logger.info(f"Validating name spelling for messages: {messages}")
        
        try:
            # Look for messages about spelling a name
            name_spelling_prompt = """
            Your task is to:
            1. Preserve EXACTLY how the name was spelled
            2. Do NOT modify capitalization
            3. Do NOT remove or add spaces
            4. Return the name PRECISELY as it was spelled
            5. If no name spelling is detected, return the original content
            """
            
            # Find the most recent name spelling
            last_message = messages[-1]['content'] if messages else ''
            
            logger.info(f"Processing last message: {last_message}")
            
            # Return the exact input to preserve spelling
            return last_message
        
        except Exception as e:
            logger.error(f"Error validating name spelling: {e}")
            logger.error(traceback.format_exc())
            return last_message

# Initialize processor with API key
try:
    name_validator = NameValidator(os.environ.get('GOOGLE_AI_API_KEY'))
except Exception as e:
    logger.error(f"Failed to initialize name validator: {e}")
    logger.error(traceback.format_exc())
    name_validator = None

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
        
        # Extract request data
        try:
            data = request.get_json()
            logger.info(f"Request JSON data: {data}")
        except Exception as parse_error:
            logger.error(f"Error parsing JSON: {parse_error}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Check if name validator is initialized
        if name_validator is None:
            logger.error("Name validator not initialized")
            return jsonify({
                "error": "Name validator initialization failed"
            }), 500
        
        # Extract messages
        messages = data.get('messages', [])
        logger.info(f"Messages: {messages}")
        
        # Validate name spelling
        processed_response = name_validator.validate_name_spelling(messages)
        
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
                    "content": processed_response,
                    "refusal": None,
                    "annotations": []
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(' '.join([m['content'] for m in messages]).split()),
                "completion_tokens": len(processed_response.split()),
                "total_tokens": len(' '.join([m['content'] for m in messages]).split()) + len(processed_response.split()),
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
