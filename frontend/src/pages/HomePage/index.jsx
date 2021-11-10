import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import { makeStyles, styled } from '@mui/styles';

import {
  scoreSequencesUrl,
  scoreSequencesFileUrl,
} from '../../routes';

const useStyles = makeStyles({
  inputSequenceTitleBox: {
    paddingTop: '30px',
    textAlign: 'center',
  },
  centered: {
    textAlign: 'center',
    justifyContent: 'center'
  },
});

export default function HomePage() {
  const classes = useStyles();
  const navigate = useNavigate();
  const [inputSequence, setInputSequence] = useState('');
  const [inputSequenceFile, setInputSequenceFile] = useState(null);
  const [resultsId, setResultsId] = useState('');

  const sendSequenceButtonOnClick = async () => {
    const scoreSequenceURLWithQuery = scoreSequencesUrl + `?sequences=${JSON.stringify(inputSequence)}`;
    const fetchedResultsId = await fetch(scoreSequenceURLWithQuery)
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        alert('Error uploading sequence, please try again.')
      }
    })
    .then((results) => {
      return results['identifier'];
    });
    setResultsId(fetchedResultsId);
  }

  const selectFileButtonOnClick = async (event) => {
		setInputSequenceFile(event.target.files[0]);
  }

  const uploadFileButtonOnClick = async (event) => {
    if (!inputSequenceFile) {
      alert('Please choose a file of sequences to score first.');
      return;
    }
    const formData = new FormData();
    formData.append('sequences_file', inputSequenceFile);
    const fetchedResultsId = await fetch(scoreSequencesFileUrl, {
      method: 'POST',
      body: formData,
    })
    .then((response) => {
      if (response.ok) {
        return response.json()
      } else {
        alert('Error uploading sequence, please try again.')
      }
    })
    .then((results) => {
      return results['identifier'];
    });
    setResultsId(fetchedResultsId);
  }

  const goToResultsPage = async () => {
    navigate(`/results/${resultsId}`);
  }

  const Input = styled('input')({
    display: 'none',
  });

  return (
    <Container>
      <Grid container style={{paddingTop: '2em', justifyContent: 'center'}}>
        <Grid item xs={12} sm={12} md={5} lg={4} xl={4}>
          <Box className={classes.inputSequenceTitleBox}>
            <h2>Input sequences</h2>
          </Box>
          <Box
            component='form'
            sx={{
              '& > :not(style)': { m: 1, width: '25ch' },
            }}
            noValidate
            className={classes.centered}
          >
            <TextField
              id='standard-basic'
              label='Sequences'
              variant='standard'
              multiline
              onChange={(e) => setInputSequence(e.target.value)}
            />
            {
              inputSequence
              ?
              <Button
                variant='contained'
                disableElevation
                onClick={sendSequenceButtonOnClick}
              >
                Score Sequences
              </Button>
              :
              <Button
                variant='outlined'
                disabled
              >
                Score Sequences
              </Button>
            }
          </Box>
        </Grid>
        <Grid item xs={0} sm={0} md={0} lg={1} xl={1}>
        </Grid>
        <Grid item xs={12} sm={12} md={5} lg={4} xl={4} className={classes.centered}>
          <Grid container spacing={1} className={classes.centered}>
            <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
              <Box className={classes.inputSequenceTitleBox}>
                <h2>Upload a file of sequences</h2>
              </Box>
            </Grid>
            <Grid item className={classes.centered}>
              <label htmlFor='contained-button-file'>
                <Input id='contained-button-file' multiple type='file' onChange={selectFileButtonOnClick}/>
                <Button variant='outlined' component='span'>
                  Choose File
                </Button>
              </label>
            </Grid>
            <Grid item className={classes.centered}>
              {
                inputSequenceFile
                ?
                <Button
                  variant='contained'
                  disableElevation
                  onClick={uploadFileButtonOnClick}
                >
                  Score Sequences File
                </Button>
                :
                <Button
                  variant='outlined'
                  disabled
                >
                  Score Sequences File
                </Button>
              }
            </Grid>
            {
              inputSequenceFile &&
              <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
                <Box>
                  <span>
                    <b>Selected:</b> {`${inputSequenceFile.name}`}
                  </span>
                </Box>
              </Grid>
            }
          </Grid>
        </Grid>
      </Grid>

      <Box className={classes.centered}>
        {
          resultsId &&
          <Grid container spacing={2} style={{paddingTop: '3em'}}>
            <Grid item xs={12}>
              <p>
                <b>Your Results ID is: </b>{resultsId}
                <br/>
                Click on the link below or save this ID to view your results later.
              </p>

              <Button
                variant='contained'
                disableElevation
                onClick={goToResultsPage}
              >
                View Results
              </Button>
            </Grid>
          </Grid>
        }
      </Box>
    </Container>
  );
}
