from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from app.core.logger import get_logger

logger = get_logger(__name__)

class ReceiptMarkServiceInterface(ABC):
    @abstractmethod
    async def detect_marks(self, image_data: bytes) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def batch_detect_marks(self, image_data_list: List[bytes]) -> List[Dict[str, Any]]:
        pass


class MobileNetV3ReceiptMarkService(ReceiptMarkServiceInterface):

    _instance: Optional["MobileNetV3ReceiptMarkService"] = None
    _model: Optional[torch.nn.Module] = None
    _transform: Optional[transforms.Compose] = None
    _device: Optional[str] = None
    _initialized: bool = False

    STAMP_THRESHOLD = 0.6
    SIGNATURE_THRESHOLD = 0.6

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

            if getattr(settings, "stamp_force_cpu", False):
                self._device = "cpu"
            elif getattr(settings, "stamp_device", None):
                device_lower = settings.stamp_device.lower()
                if device_lower in ("cuda", "gpu"):
                    self._device = self._get_safe_device()
                else:
                    self._device = "cpu"
            else:
                self._device = "cpu"

            logger.info(f"[ReceiptMarks] Using device: {self._device}")

            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            backbone = mobilenet_v3_large(weights=weights)

            backbone.classifier = torch.nn.Sequential(
                torch.nn.Linear(backbone.classifier[0].in_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 2),
                torch.nn.Sigmoid()
            )

            model_path = getattr(settings, "receipt_marks_model_path", None) or getattr(settings, "stamp_model_path", None)
            if model_path:
                logger.info(f"[ReceiptMarks] Loading fine-tuned weights from: {model_path}")
                state = torch.load(model_path, map_location=self._device)
                backbone.load_state_dict(state)
            else:
                logger.warning(
                    "[ReceiptMarks] Using ImageNet weights only – fine-tuning recommended"
                )

            backbone.to(self._device)
            backbone.eval()
            torch.set_num_threads(4)

            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self._model = backbone
            self._initialized = True

            logger.info("[ReceiptMarks] Model initialized successfully")

        except Exception:
            logger.error("Failed to initialize receipt mark detection model", exc_info=True)
            raise

    def _get_safe_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            _ = torch.tensor([1.0], device="cuda") * 2
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            return "cpu"

    def _load_image(self, image_data: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Optional but recommended: focus on bottom area of receipt
        w, h = image.size
        image = image.crop((0, int(h * 0.4), w, h))

        return image

    def _infer(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            output = self._model(input_tensor).squeeze(0)

        stamp_prob = float(output[0])
        signature_prob = float(output[1])

        return {
            "stamp": {
                "present": stamp_prob >= self.STAMP_THRESHOLD,
                "confidence": round(stamp_prob, 4),
            },
            "signature": {
                "present": signature_prob >= self.SIGNATURE_THRESHOLD,
                "confidence": round(signature_prob, 4),
            },
        }

    async def detect_marks(self, image_data: bytes) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Receipt mark detection service not initialized")

        try:
            image = self._load_image(image_data)
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            result = self._infer(tensor)

            logger.info(
                "[ReceiptMarks] "
                f"stamp={result['stamp']} | signature={result['signature']}"
            )

            return {
                "status": "success",
                "result": result,
                "device_used": self._device,
            }

        except Exception as e:
            logger.error("Receipt mark detection failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "result": None,
            }

    async def batch_detect_marks(
        self,
        image_data_list: List[bytes],
    ) -> List[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Receipt mark detection service not initialized")

        try:
            images = [self._load_image(d) for d in image_data_list]
            tensors = torch.stack([self._transform(img) for img in images]).to(self._device)

            with torch.no_grad():
                outputs = self._model(tensors)

            results = []
            for out in outputs:
                stamp_prob = float(out[0])
                signature_prob = float(out[1])

                results.append({
                    "status": "success",
                    "stamp": {
                        "present": stamp_prob >= self.STAMP_THRESHOLD,
                        "confidence": round(stamp_prob, 4),
                    },
                    "signature": {
                        "present": signature_prob >= self.SIGNATURE_THRESHOLD,
                        "confidence": round(signature_prob, 4),
                    },
                })

            logger.info(f"[ReceiptMarks] Batch processed {len(results)} images")
            return results

        except Exception as e:
            logger.error("Batch receipt mark detection failed", exc_info=True)
            return [{
                "status": "error",
                "error": str(e),
                "stamp": None,
                "signature": None,
            } for _ in image_data_list]


def get_receipt_mark_service() -> ReceiptMarkServiceInterface:
    return MobileNetV3ReceiptMarkService()


class YOLOTransformerStampService(ReceiptMarkServiceInterface):
    """
    YOLO-based signature detection and Transformer-based stamp detection service.
    Uses YOLOv8 for signatures and Ooredoo transformer model for stamps.
    """
    
    _instance: Optional["YOLOTransformerStampService"] = None
    _sig_model: Optional[Any] = None
    _stamp_model: Optional[Any] = None
    _stamp_processor: Optional[Any] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_models()
    
    def _initialize_models(self):
        if self._initialized:
            return
        
        try:
            from app.core.config import settings
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download, login
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            
            logger.info("[YOLOTransformerStamp] Initializing models...")
            
            huggingface_token = getattr(settings, "huggingface_token", None)
            if huggingface_token:
                login(token=huggingface_token)
            
            device = getattr(settings, "stamp_device", "cpu")
            if getattr(settings, "stamp_force_cpu", False):
                device = "cpu"
            
            logger.info("[YOLOTransformerStamp] Loading signature YOLO model...")
            sig_model_path = hf_hub_download(
                repo_id="tech4humans/yolov8s-signature-detector",
                filename="yolov8s.pt",
                token=huggingface_token
            )
            self._sig_model = YOLO(sig_model_path)
            self._sig_model.to(device)
            logger.info("[YOLOTransformerStamp] Signature model loaded")
            
            logger.info("[YOLOTransformerStamp] Loading stamp transformer model...")
            self._stamp_processor = AutoImageProcessor.from_pretrained(
                "Ooredoo-Group/ooredoo-stamp-detection",
                token=huggingface_token
            )
            self._stamp_model = AutoModelForObjectDetection.from_pretrained(
                "Ooredoo-Group/ooredoo-stamp-detection",
                token=huggingface_token
            ).eval()
            if device == "cuda" and torch.cuda.is_available():
                self._stamp_model.to(device)
            logger.info("[YOLOTransformerStamp] Stamp model loaded")
            
            self._initialized = True
            logger.info("[YOLOTransformerStamp] All models initialized successfully")
            
        except Exception as e:
            logger.error(f"[YOLOTransformerStamp] Failed to initialize models: {e}", exc_info=True)
            raise
    
    async def detect_marks(self, image_data: bytes) -> Dict[str, Any]:
        """Detect marks using YOLO and Transformer models."""
        if not self._initialized:
            raise RuntimeError("YOLOTransformerStamp service not initialized")
        
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._detect_marks_sync,
                image_data
            )
            
            from app.core.config import settings
            return {
                "status": "success",
                "result": result,
                "device_used": getattr(settings, "stamp_device", "cpu")
            }
            
        except Exception as e:
            logger.error("YOLOTransformerStamp detection failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "result": None,
            }
    
    def _detect_marks_sync(self, image_data: bytes) -> Dict[str, Any]:
        """Synchronous detection (runs in executor)."""
        import cv2
        import torch
        import numpy as np
        from PIL import Image
        import tempfile
        import os
        
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image_bgr)
        
        try:
            signature_detections = []
            stamp_detections = []
            
            sig_results = self._sig_model.predict(
                source=tmp_path,
                conf=0.25,
                device="cpu",
                verbose=False
            )[0]
            
            if sig_results.boxes is not None:
                for box in sig_results.boxes:
                    signature_detections.append({
                        "bbox_xyxy": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0])
                    })
            
            inputs = self._stamp_processor(images=image_pil, return_tensors="pt")
            device = next(self._stamp_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                stamp_outputs = self._stamp_model(**inputs)
            
            target_sizes = torch.tensor([image_pil.size[::-1]])
            processed = self._stamp_processor.post_process_object_detection(
                stamp_outputs,
                target_sizes=target_sizes,
                threshold=0.30
            )[0]
            
            for score, box in zip(processed["scores"], processed["boxes"]):
                stamp_detections.append({
                    "bbox_xyxy": box.tolist(),
                    "confidence": float(score)
                })
            
            has_signature = len(signature_detections) > 0
            has_stamp = len(stamp_detections) > 0
            
            sig_confidence = max([d["confidence"] for d in signature_detections], default=0.0) if has_signature else 0.0
            stamp_confidence = max([d["confidence"] for d in stamp_detections], default=0.0) if has_stamp else 0.0
            
            sig_locations = [d["bbox_xyxy"] for d in signature_detections]
            stamp_locations = [d["bbox_xyxy"] for d in stamp_detections]
            
            return {
                "stamp": {
                    "present": has_stamp,
                    "confidence": round(stamp_confidence, 4),
                    "bounding_boxes": stamp_locations
                },
                "signature": {
                    "present": has_signature,
                    "confidence": round(sig_confidence, 4),
                    "bounding_boxes": sig_locations
                }
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def batch_detect_marks(
        self,
        image_data_list: List[bytes],
    ) -> List[Dict[str, Any]]:
        """Batch detect marks."""
        results = []
        for image_data in image_data_list:
            result = await self.detect_marks(image_data)
            results.append(result)
        return results


def get_yolo_transformer_stamp_service() -> ReceiptMarkServiceInterface:
    return YOLOTransformerStampService()


# Backward compatibility aliases
def get_stamp_service() -> ReceiptMarkServiceInterface:
    from app.core.config import settings
    from app.enums.app_config import StampServiceType
    
    stamp_service_type = getattr(settings, "stamp_service_type", StampServiceType.YOLO_TRANSFORMER)
    
    if stamp_service_type == StampServiceType.YOLO_TRANSFORMER:
        return get_yolo_transformer_stamp_service()
    elif stamp_service_type == StampServiceType.IMAGE_PROCESSING:
        return get_image_processing_stamp_service()
    else:
        return get_receipt_mark_service()


# Backward compatibility interfaces
class StampServiceInterface(ReceiptMarkServiceInterface):
    pass


class MobileNetV3StampService(MobileNetV3ReceiptMarkService):
    pass


class ImageProcessingStampService(ReceiptMarkServiceInterface):

    _instance: Optional["ImageProcessingStampService"] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            logger.info("[ImageProcessingStamp] Service initialized")
    
    def _detect_signature_and_stamp(
        self,
        image_data: bytes,
        sig_min_area: int = 800,
        stamp_min_area: int = 1200,
        red_threshold: float = 1.5,
        bottom_fraction: float = 0.25
    ) -> Dict[str, Any]:
        try:
            import numpy as np
            from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label
            
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            img = np.array(image)
            
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            
            h, w = img.shape[:2]
            crop_h = int(h * bottom_fraction)
            bottom_img = img[h - crop_h:, :, :]
            
            if bottom_img.size == 0:
                return {
                    "has_signature": False,
                    "signature_confidence": 0.0,
                    "signature_locations": [],
                    "has_stamp": False,
                    "stamp_confidence": 0.0,
                    "stamp_locations": []
                }
            
            gray = np.dot(bottom_img[..., :3], [0.2989, 0.5870, 0.1140])
            gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-5) * 255
            gray = gray.astype(np.uint8)
            
            blurred = gaussian_filter(gray, sigma=1)
            thresh = np.mean(blurred) - 0.5 * np.std(blurred)
            binary = blurred < thresh
            
            binary = binary_erosion(binary, structure=np.ones((3, 3)))
            binary = binary_dilation(binary, structure=np.ones((5, 5)))
            
            labeled, num_features = label(binary)
            
            sig_locations = []
            for i in range(1, num_features + 1):
                component = (labeled == i)
                area = np.sum(component)
                if sig_min_area < area < 8000:
                    rows, cols = np.where(component)
                    if len(rows) > 0 and len(cols) > 0:
                        y_offset = h - crop_h
                        x1, y1 = int(min(cols)), int(min(rows)) + y_offset
                        x2, y2 = int(max(cols)), int(max(rows)) + y_offset
                        sig_locations.append((x1, y1, x2, y2))
            
            has_signature = len(sig_locations) > 0
            sig_confidence = min(1.0, len(sig_locations) * 0.6) if has_signature else 0.0
            
            red_channel = bottom_img[..., 0].astype(float)
            green_blue_avg = (bottom_img[..., 1] + bottom_img[..., 2]) / 2
            red_ratio = red_channel / (green_blue_avg + 1e-5)
            
            stamp_binary = (red_ratio > red_threshold) & (red_channel > 50)
            
            stamp_binary = binary_erosion(stamp_binary, structure=np.ones((3, 3)))
            stamp_binary = binary_dilation(stamp_binary, structure=np.ones((5, 5)))
            
            stamp_labeled, stamp_num = label(stamp_binary)
            stamp_locations = []
            for i in range(1, stamp_num + 1):
                component = (stamp_labeled == i)
                area = np.sum(component)
                if area > stamp_min_area:
                    rows, cols = np.where(component)
                    if len(rows) > 0 and len(cols) > 0:
                        y_offset = h - crop_h
                        x1, y1 = int(min(cols)), int(min(rows)) + y_offset
                        x2, y2 = int(max(cols)), int(max(rows)) + y_offset
                        stamp_locations.append((x1, y1, x2, y2))
            
            has_stamp = len(stamp_locations) > 0
            stamp_confidence = min(1.0, len(stamp_locations) * 0.6) if has_stamp else 0.0
            
            has_signature = has_signature and sig_confidence > 0.65
            has_stamp = has_stamp and stamp_confidence > 0.65
            
            return {
                "has_signature": has_signature,
                "signature_confidence": round(sig_confidence, 4),
                "signature_locations": sig_locations if has_signature else [],
                "has_stamp": has_stamp,
                "stamp_confidence": round(stamp_confidence, 4),
                "stamp_locations": stamp_locations if has_stamp else []
            }
            
        except Exception as e:
            logger.error(f"Image processing detection failed: {e}", exc_info=True)
            raise
    
    async def detect_marks(self, image_data: bytes) -> Dict[str, Any]:
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._detect_signature_and_stamp,
                image_data
            )
            
            return {
                "status": "success",
                "result": {
                    "stamp": {
                        "present": result["has_stamp"],
                        "confidence": result["stamp_confidence"],
                        "bounding_boxes": result.get("stamp_locations", [])
                    },
                    "signature": {
                        "present": result["has_signature"],
                        "confidence": result["signature_confidence"],
                        "bounding_boxes": result.get("signature_locations", [])
                    }
                },
                "device_used": "cpu"
            }
        except Exception as e:
            logger.error("Image processing mark detection failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "result": None,
            }
    
    async def batch_detect_marks(
        self,
        image_data_list: List[bytes],
    ) -> List[Dict[str, Any]]:
        results = []
        for image_data in image_data_list:
            result = await self.detect_marks(image_data)
            results.append(result)
        return results


def get_image_processing_stamp_service() -> ReceiptMarkServiceInterface:
    return ImageProcessingStampService()
