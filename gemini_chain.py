import asyncio
from PIL import Image
from pathlib import Path
import json
import os
from models import ModelConfig, GeminiAgent, ChainStep, ProcessingChain, ParallelStep

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "visible_amount": {"type": "string"}
                },
                "required": ["name"]
            }
        },
        "plate_size": {"type": "string"}
    },
    "required": ["ingredients"]
}

PORTION_SCHEMA = {
    "type": "object",
    "properties": {
        "total_portion_size": {"type": "string"},
        "portions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ingredient": {"type": "string"},
                    "amount": {"type": "string"},
                    "unit": {"type": "string"},
                    "grams": {"type": "number"}
                },
                "required": ["ingredient", "amount", "grams"]
            }
        }
    },
    "required": ["portions"]
}

COOKING_SCHEMA = {
    "type": "object",
    "properties": {
        "methods": {
            "type": "array",
            "items": {"type": "string"}
        },
        "added_fats": {"type": "string"}
    },
    "required": ["methods"]
}

NUTRITION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "object",
            "properties": {
                "calories": {"type": "number"},
                "protein": {"type": "number"},
                "carbs": {"type": "number"},
                "fat": {"type": "number"},
                "meal_type": {"type": "string"}
            },
            "required": ["calories", "protein", "carbs", "fat"]
        },
        "confidence": {"type": "number"}
    },
    "required": ["summary"]
}

def create_analysis_chain():
    """Create the analysis chain with optimized concurrency control"""
    # Initial Visual Analysis
    visual_agent = GeminiAgent(
        config=ModelConfig(
            temperature=0.1,
            response_structure=VISUAL_SCHEMA
        ),
        prompt_template="""Looking at this image, provide a JSON output with:
- List of all visible ingredients
- Rough visible amounts
- Overall plate/container size

Focus only on what you can directly observe in the image."""
    )

    # Portion Analysis
    portion_agent = GeminiAgent(
        config=ModelConfig(
            temperature=0.1,
            response_structure=PORTION_SCHEMA
        ),
        prompt_template="""Examining this image and using the ingredients identified: {visual_analysis},
estimate precise portions in a JSON format:
- Convert visible amounts to standard measurements
- Provide weight estimates in grams
- Consider ingredient density and volume

Base your analysis on what you see in the image and standard serving sizes."""
    )

    # Cooking Method Analysis
    cooking_agent = GeminiAgent(
        config=ModelConfig(
            temperature=0.2,
            response_structure=COOKING_SCHEMA
        ),
        prompt_template="""Looking at this image and the ingredients: {visual_analysis},
determine in JSON format:
- All visible cooking methods used
- Estimated added fats/oils

Base your analysis on visual cues in the image such as browning, texture, and appearance."""
    )

    # Final Nutrition Analysis
    nutrition_agent = GeminiAgent(
        config=ModelConfig(
            temperature=0.2,
            response_structure=NUTRITION_SCHEMA
        ),
        prompt_template="""Analyzing this image with all previous information:
Visual: {visual_analysis}
Portions: {portion_analysis}
Cooking: {cooking_analysis}

Provide JSON output with:
- Total calories
- Macronutrients in grams
- Meal type
- Confidence score (0-1)

Use the image and previous analyses to make accurate estimations."""
    )

    # Define chain with optimized parallel execution
    chain = ProcessingChain([
        ChainStep(
            visual_agent,
            "visual_analysis",
            required=True,
            retry_strategy={"max_retries": 3, "backoff_factor": 1.5}
        ),
        ParallelStep([
            ChainStep(
                portion_agent,
                "portion_analysis",
                retry_strategy={"max_retries": 2, "backoff_factor": 1},
                context_builder=lambda ctx: {
                    "visual_analysis": ctx.get("visual_analysis", {})
                }
            ),
            ChainStep(
                cooking_agent,
                "cooking_analysis",
                retry_strategy={"max_retries": 2, "backoff_factor": 1},
                context_builder=lambda ctx: {
                    "visual_analysis": ctx.get("visual_analysis", {})
                }
            )
        ]),
        ChainStep(
            nutrition_agent,
            "nutrition_analysis",
            required=True,
            retry_strategy={"max_retries": 3, "backoff_factor": 1.5},
            context_builder=lambda ctx: {
                "visual_analysis": ctx.get("visual_analysis", {}),
                "portion_analysis": ctx.get("portion_analysis", {}),
                "cooking_analysis": ctx.get("cooking_analysis", {})
            }
        )
    ])
    
    return chain

async def analyze_recipes(image_dir: str, max_concurrent: int = 5):
    """Process recipes with full pipeline parallelization"""
    # Load all image paths
    image_paths = []
    for ext in ('*.jpg', '*.png'):
        image_paths.extend(Path(image_dir).glob(ext))
    
    # Create a single chain instance
    chain = create_analysis_chain()
    
    async def process_single_image(path: Path):
        """Process a single image and save its results"""
        try:
            with Image.open(path) as img:
                # Process the image through the full pipeline
                result = await chain.process_image(img)
                
                # Save results
                output_path = path.with_suffix('.json')
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Print nutrition summary if available
                if 'nutrition_analysis' in result.get('results', {}):
                    nutrition = result['results']['nutrition_analysis'].get('data', {}).get('summary', {})
                    print(f"\nAnalysis for {path.name}:")
                    print(f"Calories: {nutrition.get('calories', 'N/A')}")
                    print(f"Protein: {nutrition.get('protein', 'N/A')}g")
                    print(f"Carbs: {nutrition.get('carbs', 'N/A')}g")
                    print(f"Fat: {nutrition.get('fat', 'N/A')}g")
                    print(f"Confidence: {result['results']['nutrition_analysis'].get('data', {}).get('confidence', 'N/A')}")
                
                return result
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return {"error": str(e), "path": str(path)}
    
    # Process all images in parallel with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_process(path):
        async with semaphore:
            return await process_single_image(path)
    
    # Run all pipelines concurrently with bounded parallelism
    tasks = [bounded_process(path) for path in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

if __name__ == "__main__":
    asyncio.run(analyze_recipes("./test_images", max_concurrent=5))