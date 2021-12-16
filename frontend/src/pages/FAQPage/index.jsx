import React from 'react';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import { makeStyles } from '@mui/styles';

import { faqPageBlurb } from '../../blurbs.jsx';

const useStyles = makeStyles({
  container: {
    maxWidth: '70%',
    paddingTop: 75,
  },
  centeredRow: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  blurbCol: {
    maxWidth: '90%',
    paddingBottom: 75,
  },
});

export default function FAQPage() {
  const classes = useStyles();
  return (
      <Container className={classes.container}>
        <Row className={ `${classes.centeredRow} ${classes.topPaddingExtra}` }>
          <Col className={ classes.blurbCol }>
            {faqPageBlurb()}
          </Col>
        </Row>
      </Container>
  );
}
