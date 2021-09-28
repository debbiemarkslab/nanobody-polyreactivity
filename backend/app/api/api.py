import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


# @router.get('/list/', response_model=List[str], tags=['Proteins'])
# async def get_protein_names(db: Session = Depends(deps.get_db)):
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/score_sequence/')
def score_new_sequence(
    sequence: str
):
    # You can do whatever you want here with the input sequence.
    # Currently, I'm just returning it as the value in a dictionary where the key
    # is 'input_sequence'
    # But you can also run any function (like the ones in score_new_sequences.py)
    # with this input sequence! And you can return a value or a file or whatever
    # from here too
    return {'input_sequence': sequence}
