import json
import uuid
import time
import random
import os
import sys
from urllib.parse import parse_qs
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import queue

# Version information
print("[API] Python version:", sys.version)
try:
    import fastapi
    print("[API] FastAPI version:", fastapi.__version__)
except:
    print("[API] FastAPI version: Unable to determine")

try:
    import uvicorn
    print("[API] Uvicorn version:", uvicorn.__version__)
except:
    print("[API] Uvicorn version: Unable to determine")

import modules.config
import modules.async_worker as worker
import modules.flags as flags
import modules.constants as constants
from modules.util import get_file_from_folder_list
from modules.private_logger import get_current_html_path

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Queue to store generation results
results_queue = {}

def parse_lora_from_query(lora_string):
    """Parse LoRA settings from query string format: name:weight,name:weight"""
    if not lora_string:
        return []
    
    loras = []
    for lora in lora_string.split(','):
        if ':' in lora:
            name, weight = lora.split(':', 1)
            try:
                weight = float(weight)
                loras.append([True, name.strip(), weight])
            except ValueError:
                loras.append([True, name.strip(), 1.0])
        else:
            loras.append([True, lora.strip(), 1.0])
    
    # Pad with empty loras to match expected count
    while len(loras) < modules.config.default_max_lora_number:
        loras.append([False, 'None', 1.0])
    
    return loras

@app.get("/")
async def root():
    """API root endpoint"""
    return JSONResponse({
        "message": "Fooocus API Server",
        "versions": {
            "python": sys.version,
            "fastapi": getattr(fastapi, '__version__', 'unknown'),
            "uvicorn": getattr(uvicorn, '__version__', 'unknown')
        },
        "endpoints": {
            "generate": "/generate",
            "status": "/status/{job_id}",
            "result": "/result/{job_id}",
            "models": "/models",
            "version": "/version"
        }
    })

@app.get("/version")
async def version_info():
    """Get version information"""
    import torch
    import gradio as gr
    
    return JSONResponse({
        "fooocus": {
            "version": getattr(modules.config, 'version', 'unknown'),
            "config": {
                "default_model": modules.config.default_base_model_name,
                "default_refiner": modules.config.default_refiner_model_name,
                "default_loras": modules.config.default_loras,
                "default_aspect_ratio": modules.config.default_aspect_ratio,
                "available_aspect_ratios": modules.config.available_aspect_ratios,
                "paths": {
                    "outputs": modules.config.path_outputs,
                    "models": modules.config.paths_checkpoints[0] if modules.config.paths_checkpoints else "unknown"
                }
            }
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "gradio": gr.__version__,
            "fastapi": getattr(fastapi, '__version__', 'unknown'),
            "uvicorn": getattr(uvicorn, '__version__', 'unknown')
        }
    })

@app.get("/generate")
async def generate_image(
    prompt: str = Query(..., description="Positive prompt for image generation"),
    negative_prompt: str = Query("", description="Negative prompt"),
    style: str = Query('Fooocus V2,Fooocus Enhance,Fooocus Sharp', description="Comma-separated list of styles"),
    performance: str = Query('Speed', description="Performance setting"),
    aspect_ratio: str = Query('', description="Aspect ratio (e.g., 1152×896)"),
    image_number: int = Query(1, description="Number of images to generate"),
    seed: int = Query(-1, description="Seed for generation, -1 for random"),
    sharpness: float = Query(2.0, description="Image sharpness"),
    guidance_scale: float = Query(4.0, description="Guidance scale"),
    base_model: str = Query("", description="Base model name (empty for default)"),
    refiner_model: str = Query("", description="Refiner model name (empty for default)"),
    refiner_switch: float = Query(0.5, description="Refiner switch point"),
    loras: str = Query("", description="LoRA settings as name:weight,name:weight"),
    sampler: str = Query("", description="Sampler name (empty for default)"),
    scheduler: str = Query("", description="Scheduler name (empty for default)"),
    output_format: str = Query('png', description="Output format (png, jpg, webp)"),
    save_metadata: bool = Query(True, description="Save metadata to images"),
    async_mode: bool = Query(False, description="Return immediately with job ID"),
    debug: bool = Query(False, description="Enable debug output")
):
    """Generate images based on provided parameters"""
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        print(f"\n[API] ========== Starting generation {job_id} ==========")
        print(f"[API] Prompt: {prompt}")
        print(f"[API] Performance: {performance}")
        
        # Use defaults if empty
        if not base_model:
            base_model = modules.config.default_base_model_name
        if not refiner_model:
            refiner_model = modules.config.default_refiner_model_name
        if not sampler:
            sampler = modules.config.default_sampler
        if not scheduler:
            scheduler = modules.config.default_scheduler
        if not negative_prompt:
            negative_prompt = modules.config.default_prompt_negative
        
        # Fix aspect ratio format
        if not aspect_ratio:
            aspect_ratio = modules.config.default_aspect_ratio
        else:
            # Convert common formats to expected format
            aspect_ratio = aspect_ratio.replace('*', '×').replace('x', '×').replace('X', '×')
        
        print(f"[API] Aspect Ratio: {aspect_ratio}")
        
        # Parse styles
        style_selections = [s.strip() for s in style.split(',') if s.strip()]
        if not style_selections:
            style_selections = modules.config.default_styles
        
        # Parse LoRAs
        lora_settings = parse_lora_from_query(loras)
        if not lora_settings:
            lora_settings = modules.config.default_loras
        
        # Use random seed if -1
        if seed == -1:
            seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        
        print(f"[API] Final seed: {seed}")
        
        # Prepare arguments for AsyncTask
        args = []
        
        # Enhance settings
        if hasattr(modules.config, 'default_enhance_tabs'):
            for i in range(modules.config.default_enhance_tabs):
                args.insert(0, False)  # enhance_mask_invert
                args.insert(0, 0)  # enhance_inpaint_erode_or_dilate
                args.insert(0, 0.618)  # enhance_inpaint_respective_field
                args.insert(0, 1.0)  # enhance_inpaint_strength
                args.insert(0, 'None')  # enhance_inpaint_engine
                args.insert(0, False)  # enhance_inpaint_disable_initial_latent
                args.insert(0, 'Inpaint or Outpaint (default)')  # enhance_inpaint_mode
                args.insert(0, 0)  # enhance_mask_sam_max_detections
                args.insert(0, 0.3)  # enhance_mask_box_threshold
                args.insert(0, 0.25)  # enhance_mask_text_threshold
                args.insert(0, 'vit_b')  # enhance_mask_sam_model
                args.insert(0, 'full')  # enhance_mask_cloth_category
                args.insert(0, 'sam')  # enhance_mask_model
                args.insert(0, '')  # enhance_negative_prompt
                args.insert(0, '')  # enhance_prompt
                args.insert(0, '')  # enhance_mask_dino_prompt_text
                args.insert(0, False)  # enhance_enabled
        
        args.insert(0, 'Original Prompts')  # enhance_uov_prompt_type
        args.insert(0, 'Before First Enhancement')  # enhance_uov_processing_order
        args.insert(0, 'Disabled')  # enhance_uov_method
        args.insert(0, False)  # enhance_checkbox
        args.insert(0, None)  # enhance_input_image
        args.insert(0, False)  # debugging_enhance_masks_checkbox
        args.insert(0, 0)  # dino_erode_or_dilate
        args.insert(0, False)  # debugging_dino
        
        # IP control settings
        if hasattr(modules.config, 'default_controlnet_image_count'):
            for i in range(modules.config.default_controlnet_image_count):
                args.insert(0, 'ImagePrompt')  # ip_type
                args.insert(0, 0.6)  # ip_weight
                args.insert(0, 0.5)  # ip_stop
                args.insert(0, None)  # ip_image
        
        # Metadata settings
        import args_manager
        if not args_manager.args.disable_metadata:
            args.insert(0, 'fooocus')  # metadata_scheme
            args.insert(0, save_metadata)  # save_metadata_to_images
        
        if not args_manager.args.disable_image_log:
            args.insert(0, False)  # save_final_enhanced_image_only
        
        # Inpaint settings
        args.insert(0, 0)  # inpaint_erode_or_dilate
        args.insert(0, False)  # invert_mask_checkbox
        args.insert(0, False)  # inpaint_advanced_masking_checkbox
        args.insert(0, 0.618)  # inpaint_respective_field
        args.insert(0, 1.0)  # inpaint_strength
        args.insert(0, 'None')  # inpaint_engine
        args.insert(0, False)  # inpaint_disable_initial_latent
        args.insert(0, False)  # debugging_inpaint_preprocessor
        
        # FreeU settings
        args.insert(0, 0.95)  # freeu_s2
        args.insert(0, 0.99)  # freeu_s1
        args.insert(0, 1.02)  # freeu_b2
        args.insert(0, 1.01)  # freeu_b1
        args.insert(0, False)  # freeu_enabled
        
        # Control settings
        args.insert(0, 0.25)  # controlnet_softness
        args.insert(0, 'joint')  # refiner_swap_method
        args.insert(0, 128)  # canny_high_threshold
        args.insert(0, 64)  # canny_low_threshold
        args.insert(0, False)  # skipping_cn_preprocessor
        args.insert(0, False)  # debugging_cn_preprocessor
        args.insert(0, False)  # mixing_image_prompt_and_inpaint
        args.insert(0, False)  # mixing_image_prompt_and_vary_upscale
        args.insert(0, -1)  # overwrite_upscale_strength
        args.insert(0, -1)  # overwrite_vary_strength
        args.insert(0, -1)  # overwrite_height
        args.insert(0, -1)  # overwrite_width
        args.insert(0, -1)  # overwrite_switch
        args.insert(0, -1)  # overwrite_step
        args.insert(0, 'Default (model)')  # vae_name
        args.insert(0, scheduler)  # scheduler_name
        args.insert(0, sampler)  # sampler_name
        args.insert(0, 2)  # clip_skip
        args.insert(0, 7.0)  # adaptive_cfg
        args.insert(0, 0.3)  # adm_scaler_end
        args.insert(0, 0.8)  # adm_scaler_negative
        args.insert(0, 1.5)  # adm_scaler_positive
        args.insert(0, False)  # black_out_nsfw
        args.insert(0, False)  # disable_seed_increment
        args.insert(0, False)  # disable_intermediate_results
        args.insert(0, False)  # disable_preview
        
        # Inpaint/outpaint settings
        args.insert(0, None)  # inpaint_mask_image
        args.insert(0, '')  # inpaint_additional_prompt
        args.insert(0, None)  # inpaint_input_image
        args.insert(0, [])  # outpaint_selections
        
        # Input image settings
        args.insert(0, None)  # uov_input_image
        args.insert(0, 'Disabled')  # uov_method
        args.insert(0, 'uov')  # current_tab
        args.insert(0, False)  # input_image_checkbox
        
        # LoRA settings
        for lora in reversed(lora_settings):
            args.insert(0, lora[2])  # lora_weight
            args.insert(0, lora[1])  # lora_model
            args.insert(0, lora[0])  # lora_enabled
        
        # Model settings
        args.insert(0, refiner_switch)  # refiner_switch
        args.insert(0, refiner_model)  # refiner_model
        args.insert(0, base_model)  # base_model
        
        # Basic settings
        args.insert(0, guidance_scale)  # guidance_scale
        args.insert(0, sharpness)  # sharpness
        args.insert(0, False)  # read_wildcards_in_order
        args.insert(0, seed)  # image_seed
        args.insert(0, output_format)  # output_format
        args.insert(0, image_number)  # image_number
        args.insert(0, aspect_ratio)  # aspect_ratios_selection
        args.insert(0, performance)  # performance_selection
        args.insert(0, style_selections)  # style_selections
        args.insert(0, negative_prompt)  # negative_prompt
        args.insert(0, prompt)  # prompt
        args.insert(0, False)  # generate_image_grid
        
        if debug:
            print(f"[API] Total arguments: {len(args)}")
            print(f"[API] Aspect ratio argument: {aspect_ratio}")
        
        # Create task
        print(f"[API] Creating AsyncTask...")
        task = worker.AsyncTask(args=args)
        print(f"[API] AsyncTask created successfully")
        
        # Store task in results queue
        results_queue[job_id] = {
            'task': task,
            'status': 'pending',
            'images': [],
            'progress': 0,
            'message': 'Task queued'
        }
        
        # Add task to worker queue
        print(f"[API] Adding task to worker queue...")
        worker.async_tasks.append(task)
        print(f"[API] Task added to queue")
        
        if async_mode:
            return JSONResponse({
                'job_id': job_id,
                'status': 'queued',
                'message': 'Image generation started'
            })
        
        # Wait for completion if not async
        timeout = time.time() + 300  # 5 minute timeout
        last_progress = -1
        
        print(f"[API] Waiting for task completion...")
        
        while time.time() < timeout:
            time.sleep(0.1)
            
            # Check for yields
            while len(task.yields) > 0:
                flag, product = task.yields.pop(0)
                
                if flag == 'preview':
                    percentage, title, _ = product
                    results_queue[job_id]['progress'] = percentage
                    results_queue[job_id]['message'] = title
                    results_queue[job_id]['status'] = 'processing'
                    if percentage != last_progress:
                        print(f"[API] Progress: {percentage}% - {title}")
                        last_progress = percentage
                    
                elif flag == 'results':
                    print(f"[API] Received results: {product}")
                    results_queue[job_id]['images'] = product
                    
                elif flag == 'finish':
                    print(f"[API] Task finished!")
                    if isinstance(product, list) and len(product) > 0:
                        print(f"[API] Generated {len(product)} images")
                        for i, img in enumerate(product):
                            print(f"[API] Image {i}: {img}")
                    
                    results_queue[job_id]['images'] = product
                    results_queue[job_id]['status'] = 'completed'
                    results_queue[job_id]['progress'] = 100
            
            # Check if completed
            if results_queue[job_id]['status'] == 'completed':
                break
            
            # Check if task has error
            if hasattr(task, 'last_stop') and task.last_stop == 'error':
                results_queue[job_id]['status'] = 'failed'
                break
        
        result = results_queue[job_id]
        
        print(f"[API] Task final status: {result['status']}")
        print(f"[API] Number of images: {len(result['images']) if result['images'] else 0}")
        
        if result['status'] == 'completed' and result['images']:
            # Return first image directly
            image_path = str(result['images'][0])
            print(f"[API] Returning image: {image_path}")
            
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"[API] Image exists, size: {file_size} bytes")
                
                # Clean up
                del results_queue[job_id]
                
                return FileResponse(
                    image_path, 
                    media_type=f'image/{output_format}',
                    headers={
                        "Content-Disposition": f"inline; filename=generated_{job_id}.{output_format}"
                    }
                )
            else:
                print(f"[API] ERROR: Image file not found at {image_path}")
        
        # If we get here, something went wrong
        error_msg = f"Image generation failed. Status: {result['status']}, Images: {len(result['images']) if result['images'] else 0}"
        print(f"[API] ERROR: {error_msg}")
        
        # Clean up
        if job_id in results_queue:
            del results_queue[job_id]
        
        raise HTTPException(status_code=500, detail=error_msg)
        
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nType: {type(e).__name__}\n{traceback.format_exc()}"
        print(f"[API] EXCEPTION: {error_details}")
        
        # Clean up on error
        if 'job_id' in locals() and job_id in results_queue:
            del results_queue[job_id]
            
        raise HTTPException(status_code=500, detail=error_details)

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check the status of an async generation job"""
    if job_id not in results_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = results_queue[job_id]
    task = result['task']
    
    # Update status from task yields
    while len(task.yields) > 0:
        flag, product = task.yields.pop(0)
        
        if flag == 'preview':
            percentage, title, _ = product
            result['progress'] = percentage
            result['message'] = title
            result['status'] = 'processing'
            
        elif flag == 'results' or flag == 'finish':
            result['images'] = product
            result['status'] = 'completed'
            result['progress'] = 100
    
    return JSONResponse({
        'job_id': job_id,
        'status': result['status'],
        'progress': result['progress'],
        'message': result['message'],
        'images': result['images'] if result['status'] == 'completed' else []
    })

@app.get("/result/{job_id}")
async def get_result(job_id: str, index: int = Query(0, description="Image index for multiple results")):
    """Get the generated image for a completed job"""
    if job_id not in results_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = results_queue[job_id]
    
    if result['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job status: {result['status']}")
    
    if not result['images'] or index >= len(result['images']):
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_path = result['images'][index]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Determine mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp'
    }.get(ext, 'image/png')
    
    return FileResponse(
        image_path, 
        media_type=mime_type,
        headers={
            "Content-Disposition": f"inline; filename=result_{job_id}_{index}{ext}"
        }
    )

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        styles_list = list(modules.config.style_keys) if hasattr(modules.config, 'style_keys') else modules.config.available_styles
    except:
        styles_list = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']
    
    # Get valid aspect ratios
    aspect_ratios = []
    for ar in modules.config.available_aspect_ratios:
        aspect_ratios.append(ar.replace('*', '×'))
    
    return JSONResponse({
        'base_models': modules.config.model_filenames,
        'refiner_models': ['None'] + modules.config.model_filenames,
        'loras': modules.config.lora_filenames,
        'vae': [flags.default_vae] + modules.config.vae_filenames,
        'styles': styles_list,
        'samplers': flags.sampler_list,
        'schedulers': flags.scheduler_list,
        'performances': flags.Performance.list(),
        'aspect_ratios': aspect_ratios,
        'default_aspect_ratio': modules.config.default_aspect_ratio
    })

def run_api_server(host='127.0.0.1', port=8888):
    """Run the API server"""
    import args_manager
    global args_manager
    print(f"[API] Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")