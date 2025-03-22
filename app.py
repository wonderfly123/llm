import json
import os
import fastapi
from fastapi.responses import StreamingResponse
from fastapi import Request
import uvicorn
import logging
import time
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve API key from environment
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
if not GOOGLE_AI_API_KEY:
    raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_AI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = fastapi.FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user_id: Optional[str] = None

class DataValidator:
    @staticmethod
    def validate_name_spelling(content):
        """
        Validate and preserve exact name spelling
        
        Args:
            content (str): Input content to validate
        
        Returns:
            dict: Validation result
        """
        # Look for specific name spelling patterns
        name_spelling_keywords = [
            "spell", "spelling", "letter by letter", 
            "spell out", "spell your name"
        ]
        
        # Check if the content suggests name spelling
        is_spelling_request = any(
            keyword in content.lower() 
            for keyword in name_spelling_keywords
        )
        
        return {
            "is_spelling_request": is_spelling_request,
            "original_content": content
        }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Process chat completion request with real-time validation
    """
    try:
        # Get the most recent message
        latest_message = request.messages[-1]
        
        # Validate the latest message
        validation_result = DataValidator.validate_name_spelling(latest_message.content)
        
        # Prepare messages for Gemini (pass through original messages)
        gemini_messages = [
            {
                'role': msg.role,
                'parts': [msg.content]
            } for msg in request.messages
        ]

        # Generate response
        response = model.generate_content(
            contents=gemini_messages,
            generation_config={
                'temperature': request.temperature or 0.7,
                'max_output_tokens': request.max_tokens or 8192
            }
        )

        # Prepare response in streaming format
        async def event_stream():
            try:
                # Create a unique ID for this completion
                completion_id = f'chatcmpl-{os.urandom(16).hex()}'
                
                # Yield the response
                yield f"data: {json.dumps({\
                    'id': completion_id,\
                    'object': 'chat.completion',\
                    'created': int(time.time()),\
                    'choices': [{\
                        'index': 0,\
                        'message': {\
                            'role': 'assistant',\
                            'content': response.text,\
                            'validation_metadata': validation_result\
                        },\
                        'finish_reason': 'stop'\
                    }]\
                })}\n\n"
                
                # End of stream marker
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return fastapi.HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
