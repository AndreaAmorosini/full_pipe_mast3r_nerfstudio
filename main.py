from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import full_pipe
import os
from fastapi.middleware.cors import CORSMiddleware
import time

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

origins = ["http://localhost", "http://localhost:8000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        video_path = ""
        output_dir = f"{lesson_dir}/images"
        frame_count = 400
        max_num_iterations = 100000
        nerfstudio_model = "splatfacto-big"
        num_downscales = 2
        
        #RUN THE FULL PIPELINE            
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                full_pipe.full_pipe(
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
        
        splat_url = ""
        
        #DELETE FOLDER
        os.rmdir(lesson_dir)
        
        #RETURN THE URL
        return Response(ply_url=splat_url)
    
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            detail=str(e),
            error_code=1001
        )
        
    
    
    
