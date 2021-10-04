from app.common import score_new_sequences

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

origins = [
    'http://localhost:3000',
    'localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/score_sequence/')
def score_new_sequence(
    sequence: str
):
    # Example of using an imported function on the input parameter "sequence"
    file_path = score_new_sequences.main(sequence)

    # Make sure the results directory exists, if not, make it
    # results_dir = '/nanobody-polyreactivity/results'
    # Path(results_dir).mkdir(parents=True, exist_ok=True)
    # file_name = 'results.txt'
    # file_path = os.path.join(results_dir, file_name)

    # with open(file_path, 'w') as f:
    #     f.write(sequence_with_dumb_prefix)

    return FileResponse(file_path, media_type='application/octet-stream')
