from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from openai import AsyncOpenAI
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class LLMServiceInterface(ABC):
    @abstractmethod
    async def extract_structured_data(
        self,
        ocr_text: str,
        stamp_detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass


class OpenRouterLLMService(LLMServiceInterface):
    _instance: Optional["OpenRouterLLMService"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.api_key = getattr(settings, "openrouter_api_key", None)
        self.model = getattr(settings, "openrouter_model", "deepseek/deepseek-r1-0528:free")
        self.base_url = getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
        self.site_url = getattr(settings, "openrouter_site_url", None)
        self.site_name = getattr(settings, "openrouter_site_name", "IDFC GenAI")
        
        if not self.api_key:
            raise ValueError("openrouter_api_key must be set in config")
        
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=extra_headers
        )
    
    async def extract_structured_data(
        self,
        ocr_text: str,
        stamp_detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured data from OCR text using LLM.
        Returns structured JSON with required fields.
        """
        prompt = self._build_prompt(ocr_text, stamp_detection)
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_dict = {
                "id": completion.id,
                "object": completion.object,
                "created": completion.created,
                "model": completion.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in completion.choices
                ],
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                    "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
                    "total_tokens": completion.usage.total_tokens if completion.usage else None
                }
            }
            
            structured_data = self._parse_response(response_dict)
            response_text = json.dumps(response_dict)
            structured_data["llm_prompt"] = prompt
            structured_data["llm_response"] = response_text
            return structured_data
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "data": None,
                "llm_prompt": prompt,
                "llm_response": None
            }
    
    def _build_prompt(self, ocr_text: str, stamp_detection: Dict[str, Any]) -> str:
        """Build the prompt for LLM extraction."""
        stamp_info = ""
        if stamp_detection.get("status") == "success":
            has_stamp = stamp_detection.get("has_stamp", False) or stamp_detection.get("stamp", {}).get("present", False)
            stamp_confidence = stamp_detection.get("stamp_confidence") or stamp_detection.get("stamp", {}).get("confidence", 0)
            has_signature = stamp_detection.get("has_signature", False) or stamp_detection.get("signature", {}).get("present", False)
            signature_confidence = stamp_detection.get("signature_confidence") or stamp_detection.get("signature", {}).get("confidence", 0)
            stamp_info = f"\n\nStamp Detection: {'Present' if has_stamp else 'Not Present'} (confidence: {stamp_confidence})\nSignature Detection: {'Present' if has_signature else 'Not Present'} (confidence: {signature_confidence})"
        
        prompt = f"""Extract the following structured information from the OCR text below and return it as valid JSON.

Required fields:
1. dealer_name (Text, fuzzy match) - Name of the dealer
2. model_name (Text, exact match) - Model name of the vehicle/tractor
3. horse_power (Numeric, exact match) - Horse power value
4. asset_cost (Numeric, exact match) - Asset cost/price
5. dealer_signature (Object with: present: boolean, bounding_box: {{x, y, width, height}} or null) - Presence of dealer signature with bounding box coordinates
6. dealer_stamp (Object with: present: boolean, bounding_box: {{x, y, width, height}} or null) - Presence of dealer stamp with bounding box coordinates

OCR Text:
{ocr_text}
{stamp_info}

Return ONLY valid JSON in this exact format, if nothing is found show null thats fine.:
{{
  "dealer_name": "string or null",
  "model_name": "string or null",
  "horse_power": number or null,
  "asset_cost": number or null,
  "dealer_signature": {{
    "present": boolean,
    "bounding_box": {{"x": number, "y": number, "width": number, "height": number}} or null
  }},
  "dealer_stamp": {{
    "present": boolean,
    "bounding_box": {{"x": number, "y": number, "width": number, "height": number}} or null
  }}
}}

Important:
- Return only the JSON object, no additional text
- Use null for missing values
- For bounding boxes, use pixel coordinates if available, otherwise null
- Extract exact values as they appear in the text"""
        
        return prompt
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenRouter response and extract JSON."""
        content = ""
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty content in response")
            
            # Try to extract JSON from response
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.strip()
            
            # Try to find JSON object
            if content.startswith("{"):
                json_end = content.rfind("}")
                if json_end > 0:
                    content = content[:json_end + 1]
            
            parsed_data = json.loads(content)
            
            return {
                "status": "success",
                "data": parsed_data
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {content[:500] if content else 'No content'}")
            return {
                "status": "error",
                "error": f"Invalid JSON in LLM response: {str(e)}",
                "data": None
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }


def get_llm_service() -> LLMServiceInterface:
    return OpenRouterLLMService()
