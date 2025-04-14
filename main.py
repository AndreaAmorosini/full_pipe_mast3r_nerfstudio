import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from full_pipe import full_pipe
from fastapi.middleware.cors import CORSMiddleware
import time
import minio
import boto3
import base64
from urllib.parse import urlparse


MINIO_EDNPOINT = "http://minio:9000"
MINIO_ROOT_USER = "minioadmin"
MINIO_ROOT_PASSWORD = "minioadmin123"
AWS_STORAGE_BUCKET_NAME = "lessons-media"

class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str, error_code: int):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        
class Response(BaseModel):
    ply_url: str | None = None

class Request(BaseModel):
    lesson_id: str | None = None
    lesson_name: str | None = None
    video_url: str | None = None
    


RETRY_LIMIT = 3
RETRY_COUNTER = 0
RETRY_COOLDOWN = 180  # seconds

app = FastAPI()

# origins = ["http://localhost", "http://localhost:8000", "*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

s3 = boto3.client(
    's3',
    endpoint_url=MINIO_EDNPOINT,
    aws_access_key_id=MINIO_ROOT_USER,
    aws_secret_access_key=MINIO_ROOT_PASSWORD
)

def extract_key_from_url(download_url: str) -> str:
    """
    Extract the S3 object key from a MinIO download URL.
    Assumes the URL follows the structure:
    http://host:port/api/v1/download-shared-object/{encoded_key}
    where {encoded_key} is a base64 encoded string.
    """
    parsed = urlparse(download_url)
    print("PARSED_URL:" + str(parsed))
    # Split the path
    path_parts = parsed.path.split("/")
    # Expecting something like ['', 'api', 'v1', 'download-shared-object', '{encoded_key}']
    encoded_key = path_parts[-1]
    print("ENCODED_KEY:" + encoded_key)
    missing_padding = len(encoded_key) % 4
    if missing_padding:
        encoded_key += '=' * (4 - missing_padding)
    try:
        key = base64.b64decode(encoded_key).decode("utf-8")
        return key
    except Exception as e:
        print(f"Decoding failed: {e}. Using the raw encoded key.")
        return encoded_key

def read_s3_file(file_name):
    try:
        #EXTRACT THE KEY FROM THE URL
        # key = extract_key_from_url(file_name)
        encoded_key = file_name.split("/")[-1]
        print("ENCODED_KEY:" + encoded_key)
        missing_padding = len(file_name.split("/")[-1]) % 4
        encoded_key = encoded_key + '=' * (4 - missing_padding) 
        video_key = base64.b64decode(encoded_key).decode("utf-8")
        print("KEY:" + video_key)       
        response = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=video_key)
        data = response['Body'].read().decode("utf-8")
        #DECODE FROM BASE64
        data = base64.b64decode(data)
        return data
    except Exception as e:
        print(f"Error reading file {file_name} from S3: {e}")
        return None
    
def write_s3_file(file_path, remote_path):
    try:
        s3.Bucket(AWS_STORAGE_BUCKET_NAME).upload_file(
            Filename=file_path,
            Key=remote_path,
        )
        print(f"File {remote_path} written to S3")
    except Exception as e:
        print(f"Error writing file {file_path} to S3: {e}")

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/extract_ply")
async def extract_ply(request: Request) -> Response:
    try:
        #CREATE A DIRECTORY FOR THE LESSON
        lesson_dir = f"/lessons/{request.lesson_name}_{request.lesson_id}"
        os.makedirs(lesson_dir, exist_ok=True)        
        #RETRIEVE THE VIDEO FROM MINIO
        video = read_s3_file(request.video_url)
        if not video:
            raise CustomHTTPException(
                status_code=404,
                detail="Video not found",
                error_code=1000
            )
        
        #SAVE THE VIDEO TO THE LESSON DIRECTORY
        video_path = f"{lesson_dir}/{request.video_url.split('/')[-1]}"
        with open(video_path, "wb") as video_file:
            video_file.write(video)
            
        output_dir = f"{lesson_dir}/images"
        frame_count = 400
        max_num_iterations = 100000
        nerfstudio_model = "splatfacto-big"
        num_downscales = 2
        
        #RUN THE FULL PIPELINE            
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                full_pipe(
                    video_path=video_path,
                    output_dir=output_dir,
                    frame_count=frame_count,
                    max_num_iterations=max_num_iterations,
                    nerfstudio_model=nerfstudio_model,
                    advanced_training = True,
                    use_mcmc = True,
                    num_downscales=num_downscales,
                )
                print("Pipeline completed successfully.")
                break  # Exit the loop if successful
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt <= RETRY_LIMIT:
                    print(f"Retrying in {RETRY_COOLDOWN} seconds...")
                    time.sleep(RETRY_COOLDOWN)
                else:
                    print("Max attempts reached. Exiting.")
                    raise e
            
        
        splat_path = f"/lessons/{request.lesson_name}_{request.lesson_id}/splat/splat.ply"
        
        #LOAD ON MINIO
        write_s3_file(splat_path, f"{request.lesson_name}_{request.lesson_id}/splat.ply")
        
        #DELETE FOLDER
        os.rmdir(lesson_dir)
        
        #RETURN THE URL
        return Response(ply_url=f"{request.lesson_name}_{request.lesson_id}/splat.ply")
    
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            detail=str(e),
            error_code=1001
        )
        
        
