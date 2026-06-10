from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import io
import json
import re
import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor
from app.core.logger import get_logger

logger = get_logger(__name__)


class OCRServiceInterface(ABC):
    @abstractmethod
    async def extract_text(
        self,
        image_data: bytes,
        task_prompt: str = "<s_receipt>"
    ) -> Dict[str, Any]:
        pass


class DonutOCRService(OCRServiceInterface):
    _instance: Optional["DonutOCRService"] = None
    _model: Optional[VisionEncoderDecoderModel] = None
    _processor: Optional[DonutProcessor] = None
    _device: Optional[str] = None
    _initialized: bool = False

    MODEL_NAME = "mychen76/invoice-and-receipts_donut_v1"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_model()

    def _initialize_model(self):
        if self._initialized:
            return

        try:
            from app.core.config import settings

            # Decide device
            if getattr(settings, "ocr_force_cpu", False):
                self._device = "cpu"
            elif getattr(settings, "ocr_device", None):
                device_lower = settings.ocr_device.lower()
                if device_lower in ["gpu", "cuda"]:
                    self._device = self._get_safe_device()
                else:
                    self._device = "cpu"
            else:
                self._device = self._get_safe_device()

            logger.info(
                f"Loading Donut OCR model: {self.MODEL_NAME} on device: {self._device}"
            )

            self._processor = DonutProcessor.from_pretrained(self.MODEL_NAME)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)

            self._model.to(self._device)
            self._model.eval()

            self._initialized = True
            logger.info("Donut OCR model loaded successfully")

        except Exception as e:
            logger.error("Failed to initialize Donut OCR model", exc_info=True)
            raise

    def _get_safe_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"

        try:
            test = torch.tensor([1.0], device="cuda")
            _ = test * 2
            del test
            torch.cuda.empty_cache()
            return "cuda"
        except Exception as e:
            logger.warning(f"CUDA test failed: {e}, using CPU")
            return "cpu"

    async def extract_text(
        self,
        image_data: bytes,
        task_prompt: str = "<s_receipt>"
    ) -> Dict[str, Any]:
        if not self._initialized or not self._model or not self._processor:
            raise RuntimeError("OCR service not initialized")

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = self._preprocess_image(image)
            logger.info(f"[OCR] Preprocessed image: {image}")

            pixel_values = self._processor(
                image,
                return_tensors="pt"
            ).pixel_values.to(self._device)

            decoder_input_ids = self._processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=1024,
                    early_stopping=True,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                    eos_token_id=self._processor.tokenizer.eos_token_id,
                    bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True
                )

            decoded = self._processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )[0]
            logger.info(f"[OCR] Decoded Raw text: {decoded}")

            decoded = decoded.replace(task_prompt, "").strip()

            try:
                parsed_result = json.loads(decoded)
            except json.JSONDecodeError:
                # TODO: TEMPORARY FIX - This XML parsing is model-specific and will break if model is switched
                # This is a workaround for the current model outputting XML-like structure instead of JSON
                # This needs to be refactored to handle different model output formats properly
                logger.warning("Failed to parse JSON, attempting XML to JSON conversion")
                try:
                    parsed_result = self._xml_to_json(decoded)
                except Exception as e:
                    logger.warning(f"XML to JSON conversion failed: {e}, returning raw text")
                    parsed_result = decoded

            logger.info("[OCR] Receipt extracted successfully")
            logger.info(f"[OCR] Extracted text: {parsed_result}")

            return {
                "status": "success",
                "raw_output": parsed_result,
                "device_used": self._device
            }

        except RuntimeError as e:
            if "cuda" in str(e).lower():
                logger.warning("CUDA error detected, retrying on CPU", exc_info=True)
                return await self._extract_text_with_cpu_fallback(image_data, task_prompt, str(e))
            raise

        except Exception as e:
            logger.error("OCR extraction failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "raw_output": None
            }

    async def _extract_text_with_cpu_fallback(
        self,
        image_data: bytes,
        task_prompt: str,
        original_error: str
    ) -> Dict[str, Any]:
        try:
            self._device = "cpu"
            self._model.to("cpu")

            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = self._preprocess_image(image)

            pixel_values = self._processor(
                image,
                return_tensors="pt"
            ).pixel_values

            decoder_input_ids = self._processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=1024
                )

            decoded = self._processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            decoded = decoded.replace(task_prompt, "").strip()

            try:
                parsed = json.loads(decoded)
            except json.JSONDecodeError:
                # TODO: TEMPORARY FIX - This XML parsing is model-specific and will break if model is switched
                # This is a workaround for the current model outputting XML-like structure instead of JSON
                # This needs to be refactored to handle different model output formats properly
                logger.warning("Failed to parse JSON, attempting XML to JSON conversion")
                try:
                    parsed = self._xml_to_json(decoded)
                except Exception as e:
                    logger.warning(f"XML to JSON conversion failed: {e}, returning raw text")
                    parsed = decoded

            logger.info("OCR succeeded on CPU fallback")

            return {
                "status": "success",
                "raw_output": parsed,
                "device_used": "cpu",
                "fallback_used": True,
                "original_error": original_error
            }

        except Exception as e:
            logger.error("CPU fallback OCR failed", exc_info=True)
            return {
                "status": "error",
                "error": f"CUDA error: {original_error}, CPU error: {str(e)}",
                "raw_output": None
            }

    @staticmethod
    def _preprocess_image(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_size = 1280
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return image

    @staticmethod
    def _xml_to_json(xml_str: str) -> Dict[str, Any]:
        """
        TODO: TEMPORARY FIX - Model-specific XML to JSON converter
        This will break if the model is switched to one that outputs different XML structure.
        Generic XML to JSON converter - parses any XML-like tags without hardcoding field names.
        Handles nested structures and arrays separated by <sep/>.
        Example: <s_store_name>value</s_store_name> -> {"store_name": "value"}
        Includes all fields even if empty - no filtering or omission.
        """
        def parse_xml_content(content: str) -> Any:
            """Recursively parse XML content, handling nested tags and arrays."""
            content = content.strip()
            if not content:
                return ""
            
            result = {}
            
            # Find all top-level tags using a simple approach
            # Pattern: <s_tag_name>content</s_tag_name>
            pos = 0
            while pos < len(content):
                # Find opening tag
                open_match = re.search(r'<s_([^>]+)>', content[pos:])
                if not open_match:
                    break
                
                tag_name = open_match.group(1)
                tag_start = pos + open_match.end()
                
                # Find matching closing tag (handle nested tags of same name)
                depth = 1
                search_pos = tag_start
                tag_end = None
                
                while search_pos < len(content):
                    # Look for opening tags with same name
                    next_open = re.search(r'<s_' + re.escape(tag_name) + r'>', content[search_pos:])
                    # Look for closing tags with same name
                    next_close = re.search(r'</s_' + re.escape(tag_name) + r'>', content[search_pos:])
                    
                    if next_close:
                        close_pos = search_pos + next_close.start()
                        if next_open and (search_pos + next_open.start()) < close_pos:
                            # Found nested opening tag first
                            depth += 1
                            search_pos = search_pos + next_open.end()
                        else:
                            # Found closing tag
                            depth -= 1
                            if depth == 0:
                                tag_end = close_pos
                                break
                            search_pos = close_pos + next_close.end()
                    elif next_open:
                        depth += 1
                        search_pos = search_pos + next_open.end()
                    else:
                        break
                
                if tag_end is None:
                    # No matching closing tag, skip
                    pos = tag_start
                    continue
                
                # Extract tag content
                tag_content = content[tag_start:tag_end]
                
                # Check if content contains <sep/> (array separator)
                if '<sep/>' in tag_content:
                    # Split by separator and parse each item
                    items = re.split(r'<sep/>', tag_content)
                    parsed_items = []
                    for item in items:
                        item = item.strip()
                        if item:
                            parsed_item = parse_xml_content(item)
                            if isinstance(parsed_item, dict):
                                parsed_items.append(parsed_item)
                            else:
                                parsed_items.append({"value": parsed_item} if parsed_item else {})
                    result[tag_name] = parsed_items if parsed_items else []
                else:
                    # Parse nested content
                    parsed_content = parse_xml_content(tag_content)
                    
                    # If tag already exists, convert to array
                    if tag_name in result:
                        if not isinstance(result[tag_name], list):
                            result[tag_name] = [result[tag_name]]
                        result[tag_name].append(parsed_content)
                    else:
                        result[tag_name] = parsed_content
                
                pos = tag_end + len(f'</s_{tag_name}>')
            
            # If no tags found but content exists, return as string
            if not result and content:
                return content
            
            return result if result else ""
        
        parsed = parse_xml_content(xml_str)
        
        # Ensure result is a dict
        if isinstance(parsed, dict):
            return parsed
        else:
            return {"content": parsed}


def get_donut_ocr_service() -> OCRServiceInterface:
    return DonutOCRService()
