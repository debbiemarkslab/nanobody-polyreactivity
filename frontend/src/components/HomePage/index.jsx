import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import { makeStyles } from '@mui/styles';

const useStyles = makeStyles({
  inputSequenceTitleBox: {
    paddingTop: '30px',
    textAlign: 'center',
  },
  inputSequenceBox: {
    paddingTop: '10px',
    textAlign: 'center',
  },
  results: {
    textAlign: 'center',
  },
});

const baseURL = "http://localhost:8080/";
const scoreSequenceURL = baseURL + "score_sequence/";

export default function HomePage() {
  const classes = useStyles();
  const [inputSequence, setInputSequence] = useState('');
  const [result, setResult] = useState('');

  const handleOnClick = async () => {
    const scoreSequenceURLWithQuery = scoreSequenceURL + `?sequence=${inputSequence}`;
    const fetchedResult = await fetch(scoreSequenceURLWithQuery)
    .then((response) => {
      return response.json();
    });
    setResult(JSON.stringify(fetchedResult));
  }

  return (
    <>
      <Box className={classes.inputSequenceTitleBox}>
        <h2>Input Sequence Here</h2>
      </Box>

      <Box
        component="form"
        sx={{
          '& > :not(style)': { m: 1, width: '25ch' },
        }}
        noValidate
        className={classes.inputSequenceBox}
      >
        <TextField
          id="standard-basic"
          label="Sequence"
          variant="standard"
          onChange={(e) => setInputSequence(e.target.value)}
        />
        <Button
          variant="outlined"
          onClick={handleOnClick}
        >
          Trigger API
        </Button>
      </Box>
      
      <Box className={classes.inputSequenceBox}>
        {
          result && <p className={classes.results}>The server returned: {result}</p>
        }
      </Box>
    </>
  );
}
