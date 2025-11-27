import asyncio
from werkzeug.utils import secure_filename
from apiflask import APIFlask, Schema, abort
from apiflask.fields import Integer, String, Number
from apiflask.validators import Length, OneOf
from schemas.Image import Image
from schemas.DetectObject import DetectObject
from schemas.CreateJobRequest import CreateJobRequest
from schemas.JobMetadata import JobMetadata
from schemas.Task import Task
from ml.detector import process_image
from concurrent.futures import ThreadPoolExecutor
from marshmallow import ValidationError
from ml.video_handler import process_video_task
from schemas.Result import Result
import os
import time
import json
import uuid

app = APIFlask(__name__)
executor = ThreadPoolExecutor(max_workers=2)
JOBS = {}
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

@app.post('/detect')
@app.input(Image, location='files')
@app.output(DetectObject(many=True))
async def detect_obj(files_data):
    f = files_data['image']
    filename = secure_filename(f.filename)
    f.save(os.path.join("./files", filename))
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(executor, process_image, f"./files/{filename}")
    return results


def job_wrapper(job_id, pixel_size, filepath, selected_time, detect_obj):
    try:
        JOBS[job_id]['status'] = 'processing'
        result_data = process_video_task(filepath, pixel_size, selected_time, detect_obj)
        JOBS[job_id]['data'] = result_data
        JOBS[job_id]['status'] = 'done'
    except:
        print(f"Job {job_id} failed: {e}")
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error_message'] = str(e)
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass


@app.post("/jobs")
@app.input(CreateJobRequest, location="files")
@app.output(Task)
async def create_job(files_data):
    video_file = files_data['video_file']
    json_str = files_data['data']
    job_id = str(uuid.uuid4())
    current_time = int(time.time())
    filename = f"{job_id}_{secure_filename(video_file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(filepath)
    try:
        raw_meta = json.loads(json_str)
        meta = JobMetadata().load(raw_meta)
        
        JOBS[job_id] = {
            'id': job_id,
            'created_at': current_time,
            'status': 'pending',
            'data': None
        }
        selected_time = meta['selected_frame_time']
        detect_obj = meta['detect_object']
        pixel_size = meta['pixel_size']
        print(detect_obj)

        executor.submit(job_wrapper, job_id, pixel_size, filepath, selected_time, detect_obj)
        # executor.submit(
        #     process_video_task, 
        #     job_id, 
        #     filepath, 
        #     1, # пиксель сайз??
        #     selected_time, 
        #     detect_obj
        # )

        return {'id': job_id, 'created_at': current_time}
    except json.JSONDecodeError:
        return {'message': 'Invalid JSON format in "data" field'}, 400
    except ValidationError:
        return {'message': 'Validation error'}, 422


@app.get('/jobs/<id>')
@app.output(Result)
def get_job_result(id):
    job = JOBS.get(id)
    if not job:
        return {'status': 'error', 'error_message': 'Job not found'}, 404
        
    response = {
        'status': job['status'],
        'data': job.get('data', {}),
        'error_message': job.get('error_message')
    }
    return response


@app.get('/')
async def say_hello():
    await asyncio.sleep(1)
    return {"message": "hello"}