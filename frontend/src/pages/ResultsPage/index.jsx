import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import { makeStyles } from '@mui/styles';

import { scoresUrl } from '../../routes';
import ResultsTable from '../../components/ResultsTable';

const useStyles = makeStyles({
  heading: {
    paddingTop: '30px'
  },
  container: {
    maxWidth: '90%'
  },
  centered: {
    justifyContent: 'center'
  },
  downloadButton: {
    textAlign: 'center',
    justifyContent: 'center',
    paddingTop: '2em',
  },
  table: {
    justifyContent: 'center',
    overflowX: 'scroll',
  },
});

export default function ResultsPage() {
  const classes = useStyles();
  const navigate = useNavigate();
  const { resultsId } = useParams();
  const [resultsIdInput, setResultsIdInput] = useState('');
  const [resultsStr, setResultsStr] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');

  useEffect(() => {
    const getScoresForId = async () => {
      if (!resultsId) {
        return;
      }
      const scoreSequenceURLWithQuery = scoresUrl(resultsId);
      const fetchedResultBlob = await fetch(scoreSequenceURLWithQuery)
      .then((response) => {
        if (response.ok) {
          return response.blob();
        } else {
          alert('Your result ID is either invalid or your results are not ready yet.')
        }
      });
      if (!fetchedResultBlob) {
        return;
      }
      const url = window.URL.createObjectURL(fetchedResultBlob);
      setDownloadUrl(url);
      const resultText = await fetchedResultBlob.text();
      setResultsStr(`${resultText}`);
    }
    getScoresForId();
  }, [resultsId]);

  const showResults = () => {
    return (
      <Grid container style={{paddingTop: '2em'}}>
        <Grid item className={classes.table} xs={12}>
          <ResultsTable resultsCSVStr={resultsStr}/>
        </Grid>
        <Grid item className={classes.downloadButton} xs={12}>
          {
            downloadUrl &&
            <Button
              variant="outlined"
              href={downloadUrl}
              download={`${resultsId}.csv`}
            >
              Download Results
            </Button>
          }
        </Grid>
      </Grid>
    )
  };

  const showResultsSearchBar = () => {
    const retrieveResults = () => {
      navigate(`/results/${resultsIdInput}`)
    };
    return (
      <Grid container className={classes.centered}>
        <Box
          component='form'
          sx={{
            '& > :not(style)': { m: 1, width: '30ch' },
          }}
          noValidate
        >
          <TextField
            id='standard-basic'
            label='Results ID'
            variant='standard'
            multiline
            onChange={(e) => setResultsIdInput(e.target.value)}
          />
          {
            resultsIdInput
            ?
            <Button
              variant='outlined'
              disableElevation
              onClick={retrieveResults}
            >
              Retrieve Results
            </Button>
            :
            <Button
              variant='outlined'
              disabled
            >
              Retrieve Results
            </Button>
          }
        </Box>
      </Grid>
    )
  }

  return (
    <Container className={classes.container}>
      <Grid container style={{paddingTop: '2em'}}>
        <Grid item xs={12} sm={12} md={5} lg={4} xl={4}>
          <Box className={classes.heading}>
            <h2>Results</h2>
          </Box>
        </Grid>
        {
          showResultsSearchBar()
        }
        {
          resultsId &&
          showResults()
        }
      </Grid>
    </Container>
  );
}
