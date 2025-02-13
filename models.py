from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Union
import os
import json
import io
from PIL import Image
import asyncio
from functools import partial
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image as VertexImage, GenerationConfig
from dotenv import load_dotenv
from collections import OrderedDict

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for Gemini model"""
    model_name: str = 'gemini-1.5-flash'
    temperature: float = 0.2
    max_output_tokens: int = 2048
    response_structure: Optional[Dict] = None
    
    def to_generation_config(self) -> GenerationConfig:
        """Create a GenerationConfig with structured output if schema is provided"""
        config_params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": 1,
        }
        
        if self.response_structure:
            config_params.update({
                "response_mime_type": "application/json",
                "response_schema": self.response_structure
            })
        
        return GenerationConfig(**config_params)

class GeminiAgent:
    """Enhanced Gemini model with structured prompt handling using Vertex AI"""
    # Use a BoundedSemaphore for stricter concurrency control
    _rate_limit_semaphore = asyncio.BoundedSemaphore(1000)
    
    def __init__(self, config: ModelConfig, prompt_template: str, output_processor: Optional[Callable] = None):
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
            
        vertexai.init(
            project=project_id, 
            location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        )
        
        self.config = config
        self.prompt_template = prompt_template
        self.output_processor = output_processor or (lambda x: x)
        self.model = GenerativeModel(config.model_name)
        
        # Use OrderedDict for LRU cache implementation
        self._image_cache = OrderedDict()
        self._cache_size = 1000
    
    async def _get_cached_image(self, image: Image.Image) -> bytes:
        """Async image cache with LRU implementation"""
        image_hash = hash(image.tobytes())
        
        if image_hash in self._image_cache:
            # Move to end (most recently used)
            self._image_cache.move_to_end(image_hash)
            return self._image_cache[image_hash]
            
        # Process new image
        img_byte_arr = io.BytesIO()
        await asyncio.to_thread(
            image.save,
            img_byte_arr,
            format='JPEG',
            quality=85,
            optimize=True
        )
        
        if len(self._image_cache) >= self._cache_size:
            self._image_cache.popitem(last=False)  # Remove least recently used
            
        self._image_cache[image_hash] = img_byte_arr.getvalue()
        return self._image_cache[image_hash]
    
    async def process(self, image: Image.Image, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process image with improved error handling and retries"""
        async with self._rate_limit_semaphore:
            try:
                formatted_prompt = self._format_prompt(context)
                img_bytes = await self._get_cached_image(image)
                image_obj = VertexImage.from_bytes(img_bytes)
                
                generation_config = self.config.to_generation_config()
                
                # Implement exponential backoff with jitter
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self.model.generate_content_async(
                            [formatted_prompt, image_obj],
                            generation_config=generation_config
                        )
                        return self.output_processor(self._parse_response(response))
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        # Add jitter to prevent thundering herd
                        jitter = random.uniform(0, 0.1)
                        await asyncio.sleep(2 ** attempt + jitter)
                
            except Exception as e:
                return {"error": str(e)}
    
    def _parse_response(self, response) -> Union[Dict[str, Any], str]:
        """Optimized response parsing"""
        if not response.candidates:
            return {"error": "No response candidates found"}
            
        candidate = response.candidates[0]
        text = getattr(candidate, 'text', None)
        
        if text is None and hasattr(candidate, 'parts'):
            text = next((
                part.text for part in candidate.parts 
                if hasattr(part, 'text')
            ), None)
            
        if text is None:
            return {"error": "No text found in response"}
            
        if not self.config.response_structure:
            return text
            
        try:
            text = text.strip().strip('"').replace('\\n', ' ')
            if text.startswith('\\\"'):
                text = text.encode().decode('unicode_escape')
            return json.loads(text)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def _format_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Optimized prompt formatting"""
        if not context:
            return self.prompt_template
            
        safe_context = {
            key: (
                json.dumps(value, indent=2) if isinstance(value, dict)
                else ', '.join(map(str, value)) if isinstance(value, list)
                else str(value)
            )
            for key, value in context.items()
        }
        
        return self.prompt_template.format(**safe_context)

class ChainStep:
    """Enhanced chain step with improved retry logic"""
    def __init__(self, agent: GeminiAgent, name: str, required: bool = True,
                 retry_strategy: Dict[str, Any] = None,
                 context_builder: Optional[Callable] = None):
        self.agent = agent
        self.name = name
        self.required = required
        self.retry_strategy = retry_strategy or {"max_retries": 2, "backoff_factor": 1}
        self.context_builder = context_builder or (lambda ctx: ctx)
        
    async def execute(self, image: Image.Image, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute with improved error handling"""
        processed_context = self.context_builder(context or {})
        max_retries = self.retry_strategy["max_retries"]
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.agent.process(image, processed_context)
                
                if isinstance(result, dict) and "error" in result:
                    if attempt == max_retries:
                        return {
                            "status": "skipped" if not self.required else "error",
                            "error": result["error"]
                        }
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1)
                    await asyncio.sleep(
                        self.retry_strategy["backoff_factor"] * (2 ** attempt) + jitter
                    )
                    continue
                    
                return {
                    "status": "success",
                    "data": result,
                    "metadata": {
                        "attempts": attempt + 1,
                        "step": self.name
                    }
                }
                
            except Exception as e:
                if attempt == max_retries:
                    return {
                        "status": "skipped" if not self.required else "error",
                        "error": str(e)
                    }
                jitter = random.uniform(0, 0.1)
                await asyncio.sleep(
                    self.retry_strategy["backoff_factor"] * (2 ** attempt) + jitter
                )

class ParallelStep:
    """Optimized parallel execution handler"""
    def __init__(self, steps: List[ChainStep]):
        self.steps = steps
        self.name = "parallel_steps"
        self.required = any(step.required for step in steps)
    
    async def execute(self, image: Image.Image, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with improved concurrency"""
        tasks = [
            asyncio.create_task(step.execute(image, context))
            for step in self.steps
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = {
                step.name: (
                    {"status": "error", "error": str(result)}
                    if isinstance(result, Exception)
                    else result
                )
                for step, result in zip(self.steps, results)
            }
            
            return {
                "status": "success",
                "data": processed_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Parallel execution failed: {str(e)}"
            }

class ProcessingChain:
    """Optimized processing chain with improved batch handling"""
    def __init__(self, steps: List[Union[ChainStep, ParallelStep]]):
        self.steps = steps
    
    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process single image with improved context handling"""
        context = {}
        results = {}
        
        for step in self.steps:
            try:
                step_result = await step.execute(image, context)
                
                if isinstance(step, ParallelStep):
                    if step_result.get("status") == "success":
                        results.update(step_result["data"])
                        # Update context atomically
                        context.update({
                            k: v.get("data", {})
                            for k, v in step_result["data"].items()
                            if v.get("status") == "success"
                        })
                else:
                    results[step.name] = step_result
                    if step_result.get("status") == "success":
                        context[step.name] = step_result["data"]
                
                if step.required and step_result.get("status") == "error":
                    break
                    
            except Exception as e:
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "metadata": {"step_type": "parallel" if isinstance(step, ParallelStep) else "sequential"}
                }
                
                if isinstance(step, ParallelStep):
                    results.update({s.name: error_result for s in step.steps})
                else:
                    results[step.name] = error_result
                
                if step.required:
                    break
        
        return {
            "results": results,
            "metadata": {
                "completed_steps": [
                    name for name, result in results.items()
                    if result.get("status") == "success"
                ]
            }
        }
    
    async def process_batch(self, images: List[Image.Image], max_concurrent: int = 1000) -> List[Dict[str, Any]]:
        """Optimized batch processing with adaptive concurrency"""
        results = []
        chunks = [images[i:i + max_concurrent] for i in range(0, len(images), max_concurrent)]
        
        for chunk in chunks:
            # Create tasks for the chunk
            tasks = [self.process_image(img) for img in chunk]
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
            
            # Adaptive delay based on chunk size
            if len(chunk) == max_concurrent:
                await asyncio.sleep(0.1)  # Only delay between full chunks
        
        return results