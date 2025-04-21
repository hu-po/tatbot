import argparse
import asyncio
import base64
import glob
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional, Awaitable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import aiofiles

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

@dataclass
class AIConfig:
    enabled_models: List[str] = field(default_factory=lambda: ["gpt", "claude", "gemini", "xapi", "replicate"])
    # https://docs.anthropic.com/en/docs/about-claude/models/all-models
    claude_model: str = "claude-3-7-sonnet-20250219"
    # https://platform.openai.com/docs/models
    gpt_model: str = "o3-mini-2025-01-31"
    # https://ai.google.dev/gemini-api/docs/models
    gemini_model: str = "gemini-2.5-pro-preview-03-25"
    # https://docs.x.ai/docs/models
    xai_model: str = "grok-3"
    # https://replicate.com/deepseek-ai/deepseek-r1
    replicate_model: str = "deepseek-ai/deepseek-r1"
    max_tokens: int = int(1e4)
    api_max_retries: int = 1
    timeout_model_api: int = 600 # seconds (this is high because of reasoning models)
    timeout_analysis: int = 900  # seconds

config = AIConfig()

def async_retry_decorator(*, timeout: int, max_retries: int):
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=5, max=20),  # Increased min/max wait times
            retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)),
            reraise=True,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            attempt = 1
            try:
                while True:
                    try:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        log.info(f"{func.__name__} succeeded on attempt {attempt} after {elapsed:.2f}s")
                        return result
                    except Exception as e:
                        if attempt >= max_retries:
                            raise
                        attempt += 1
                        elapsed = (datetime.now() - start_time).total_seconds()
                        log.warning(f"{func.__name__} attempt {attempt-1} failed after {elapsed:.2f}s: {type(e).__name__} - {str(e)}")
            except asyncio.TimeoutError as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                log.error(f"{func.__name__} timed out after {elapsed:.2f}s on attempt {attempt} (API timeout={timeout}s)")
                raise TimeoutError(f"{func.__name__} timed out after {timeout} seconds")
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                log.error(f"{func.__name__} failed after {elapsed:.2f}s on attempt {attempt}: {type(e).__name__} - {str(e)}")
                raise
        return wrapper
    return decorator

def encode_image(image_path: str) -> str:
    if not image_path:
        return ""
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

@async_retry_decorator(timeout=config.timeout_model_api, max_retries=config.api_max_retries)
async def async_claude(prompt: str, image_path: str = None) -> str:
    try:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        base64_image = encode_image(image_path) if image_path else ""
        client = Anthropic(api_key=api_key)
        log.info("Calling Claude API")
        log.debug(f"\n---prompt - claude\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

        response = await asyncio.to_thread(
            client.messages.create,
            model=config.claude_model,
            max_tokens=config.max_tokens,
            messages=messages,
        )
        response = response.content[0].text
        log.info("Claude API responded")
        log.debug(f"\n---reply - claude\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"Claude API error: {str(e)}")
        return f"Claude API error: {str(e)}"

@async_retry_decorator(timeout=config.timeout_model_api, max_retries=config.api_max_retries)
async def async_gpt(prompt: str, image_path: str = None) -> str:
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        base64_image = encode_image(image_path) if image_path else ""
        log.info("Calling GPT API")
        log.debug(f"\n---prompt - gpt\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.gpt_model,
            max_completion_tokens=config.max_tokens,
            messages=messages,
        )
        response = response.choices[0].message.content
        log.info("GPT API responded")
        log.debug(f"\n---reply - gpt\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"GPT API error: {str(e)}")
        return f"GPT API error: {str(e)}"

@async_retry_decorator(timeout=config.timeout_model_api, max_retries=config.api_max_retries)
async def async_xapi(prompt: str, image_path: str = None) -> str:
    try:
        from openai import OpenAI
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        base64_image = encode_image(image_path) if image_path else ""
        log.info("Calling XAI API")
        log.debug(f"\n---prompt - gpt\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.xai_model,
            max_tokens=config.max_tokens,
            messages=messages,
        )
        response = response.choices[0].message.content
        log.info("XAI API responded")
        log.debug(f"\n---reply - xai\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"XAI API error: {str(e)}")
        return f"XAI API error: {str(e)}"

@async_retry_decorator(timeout=config.timeout_model_api, max_retries=config.api_max_retries)
async def async_gemini(prompt: str, image_path: str = None) -> str:
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        uploaded_file = genai.upload_file(image_path) if image_path else None
        log.info(f"Uploaded file to Gemini: {uploaded_file.uri if uploaded_file else 'None'}")

        model = genai.GenerativeModel(model_name=config.gemini_model)
        log.debug(f"\n---prompt - gemini\n {prompt}\n---\n")
        content = [uploaded_file, "\n\n", prompt] if uploaded_file else [prompt]

        response = await model.generate_content_async(
            content,
            request_options={"timeout": config.timeout_model_api},
            generation_config={"max_output_tokens": config.max_tokens},
        )
        response = response.text
        log.info("Gemini API responded")
        log.debug(f"\n---reply - gemini\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}"

@async_retry_decorator(timeout=config.timeout_model_api, max_retries=config.api_max_retries)
async def async_replicate(prompt: str, image_path: str = None) -> str:
    try:
        import replicate
        api_key = os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("REPLICATE_API_TOKEN not set")
        log.debug(f"\n---prompt - replicate\n {prompt}\n---\n")
        image_input = None
        if image_path:
            with open(image_path, "rb") as image_file:
                image_input = image_file
                log.info(f"Loaded image for Replicate: {image_path}")
        input_data = {"prompt": prompt}
        if image_input:
            input_data["image"] = image_input
        response = await asyncio.to_thread(
            replicate.run,
            config.replicate_model,
            input=input_data
        )
        # Convert response to string if it's a generator or list
        if hasattr(response, '__iter__') and not isinstance(response, str):
            response = "".join(response)   
        log.info("Replicate API responded")
        log.debug(f"\n---reply - replicate\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"Replicate API error: {str(e)}")
        return f"Replicate API error: {str(e)}"


AI_MODEL_MAP: Dict[str, Callable[[str, Optional[str]], Awaitable[str]]] = {}
if "claude" in config.enabled_models:
    AI_MODEL_MAP["claude"] = async_claude
if "gpt" in config.enabled_models:
    AI_MODEL_MAP["gpt"] = async_gpt
if "gemini" in config.enabled_models:
    AI_MODEL_MAP["gemini"] = async_gemini
if "xapi" in config.enabled_models:
    AI_MODEL_MAP["xapi"] = async_xapi
if "replicate" in config.enabled_models:
    AI_MODEL_MAP["replicate"] = async_replicate


async def async_inference(
    prompt: str,
    image_path: str,
    enabled_models: List[str] = config.enabled_models,
) -> Dict[str, str]:
    log.debug("Starting AI analysis")
    try:
        if not config.enabled_models:
            log.error("No AI APIs enabled")
            raise ValueError("No AI APIs enabled")

        if image_path is not None and not os.path.exists(image_path):
            log.error(f"Image not found at {image_path}")
            return {"error": f"Image not found at {image_path}"}

        tasks = []
        ai_models = []

        for ai_model in enabled_models:
            tasks.append(AI_MODEL_MAP[ai_model](prompt, image_path))
            ai_models.append(ai_model)

        if not tasks:
            log.error("No tasks created - check enabled models and analyses")
            raise ValueError("No enabled models found in model map")

        log.debug(f"Executing {len(tasks)} inference tasks")
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.timeout_analysis,
            )
        except asyncio.TimeoutError:
            _msg = f"Analysis timed out after {config.timeout_analysis} seconds"
            log.error(_msg)
            return {"error": _msg}

        log.debug("Inference results:")
        results = {}
        for ai_model, resp in zip(ai_models, responses):
            results[ai_model] = resp if not isinstance(resp, Exception) else str(resp)
            if isinstance(results[ai_model], Exception):
                log.error(f"{ai_model} failed: {results[ai_model]}")
            else:
                log.debug(f"{ai_model} success: {results[ai_model]}")

        # Only save JSON results if image_path was provided
        if image_path is not None:
            try:
                # Get base filename from image path and add analysis suffix with timestamp
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                directory = os.path.dirname(image_path)
                timestamp = datetime.now().strftime("%Y%m%d%H%M")
                json_path = os.path.join(directory, f"{base_name}-analysis-{timestamp}.json")
                
                # Ensure results are JSON serializable
                serializable_results = {k: str(v) if isinstance(v, Exception) else v for k, v in results.items()}
                
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(json.dumps(serializable_results, indent=2))
                log.info(f"Saved JSON results to {json_path}")
            except Exception as e:
                log.error(f"Failed to save JSON results: {e}")

    except Exception as e:
        log.error(f"AI analysis error: {str(e)}", exc_info=True)
        return {"error": f"AI analysis failed: {str(e)}"}

    return results

def inference(prompt: str, models: List[str] = config.enabled_models, image_path: str = None) -> Dict[str, str]:
    return asyncio.run(async_inference(prompt, models, image_path))

async def async_test_model_apis() -> None:
    log.info(f"Testing enabled models: {config.enabled_models}")
    results = []
    for model_name in config.enabled_models:
        try:
            log.info(f"Testing {model_name}")
            prompt = "ur favorite emoji"
            response = await AI_MODEL_MAP[model_name](prompt)
            log.info(f"\treply:\n{response}\n")
            results.append(f"✅ {model_name.upper()}")
        except Exception as e:
            log.error(f"Error testing {model_name}: {str(e)}")
            results.append(f"❌ {model_name.upper()}: {str(e)}")
    
    # Print summary
    log.info("\n=== Model Test Results ===")
    for result in results:
        log.info(result)
    log.info("=====================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI model tests or analyze images")
    parser.add_argument("--test", action="store_true", help="Run model API tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.test:
        asyncio.run(async_test_model_apis())
    else:
        results = inference("write a short haiku about yourself")
        log.info(results)