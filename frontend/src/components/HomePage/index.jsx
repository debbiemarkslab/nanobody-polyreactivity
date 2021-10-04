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

const baseURL = "";
const scoreSequenceURL = baseURL + "/score_sequence/";

export default function HomePage() {
  const classes = useStyles();
  const [inputSequence, setInputSequence] = useState('');
  const [resultStr, setResultStr] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const resultsFileName = 'results.txt';

  const sendSequenceButtonOnClick = async () => {
    const scoreSequenceURLWithQuery = scoreSequenceURL + `?sequence=${JSON.stringify(inputSequence)}`;
    const fetchedResultBlob = await fetch(scoreSequenceURLWithQuery)
    .then((response) => {
      return response.blob();
    });
    const url = window.URL.createObjectURL(fetchedResultBlob);
    setDownloadUrl(url);
    const resultText = await fetchedResultBlob.text();
    setResultStr(`${resultText}`);
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
          multiline
          onChange={(e) => setInputSequence(e.target.value)}
        />
        <Button
          variant="outlined"
          onClick={sendSequenceButtonOnClick}
        >
          Trigger API
        </Button>
      </Box>

      <Box className={classes.inputSequenceBox}>
        {
          downloadUrl &&
          <Button
            variant="outlined"
            href={downloadUrl}
            download={resultsFileName}
          >
            Download Results
          </Button>
        }
        {
          resultStr &&
          <p>
            <b>Results have arrived! You can download them above or view the contents of the results file below:</b>
            <br/>
            {resultStr}
          </p>
        }
      </Box>
    </>
  );
}
