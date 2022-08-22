from app.common import score_new_sequences

import asyncio, os, uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
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
    allow_methods=['*'],
    allow_headers=['*']
)
@app.get('/plots/{identifier}/')
async def get_plots(identifier: str):
    plots_filepath = f'/nanobody-polyreactivity/results/plots/{identifier}.pdf'
    if not os.path.exists(plots_filepath):
        raise HTTPException(status_code=404, detail='plots file not found')
    return FileResponse(plots_filepath)

@app.get('/scores/{identifier}/')
async def get_scores_for_sequence(identifier: str):
    scores_filepath = f'/nanobody-polyreactivity/results/{identifier}_scores.csv'
    if not os.path.exists(scores_filepath):
        raise HTTPException(status_code=404, detail='Results file not found.')
    return FileResponse(scores_filepath, media_type='application/octet-stream')

@app.get('/score_sequences/')
async def score_sequences(
    sequences: str,
    doubles: bool = False,
):
    inputs_dir = '/nanobody-polyreactivity/inputs'
    Path(inputs_dir).mkdir(parents=True, exist_ok=True)

    identifier = str(uuid.uuid4())
    sequences_filepath = f'/nanobody-polyreactivity/inputs/{identifier}.fa'
    with open(sequences_filepath, 'w') as f:
        for l in sequences.strip('"').split('\\n'):
            f.write(l+'\n')

    asyncio.create_task(score_new_sequences.score_sequences(sequences_filepath, identifier, doubles))
    return {'identifier': identifier}

@app.post('/score_sequences_file/')
async def score_sequences_file(
    sequences_file: UploadFile = File(...),
    doubles: bool = False,
):
    inputs_dir = '/nanobody-polyreactivity/inputs'
    Path(inputs_dir).mkdir(parents=True, exist_ok=True)

    identifier = str(uuid.uuid4())
    sequences_filepath = f'/nanobody-polyreactivity/inputs/{identifier}.fa'
    with open(sequences_filepath, 'w') as f:
        file_bytes = await sequences_file.read()
        sequences = file_bytes.decode('utf-8')
        f.write(sequences)

    asyncio.create_task(score_new_sequences.score_sequences(sequences_filepath, identifier, doubles))
    return {'identifier': identifier}
