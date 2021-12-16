import React from 'react';
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { Link } from 'react-router-dom';
import { makeStyles } from '@mui/styles';

const useStyles = makeStyles({
  container: {
    maxWidth: '90%',
  },
});

export default function NavigationBar() {
  const classes = useStyles();

  return (
    <Navbar collapseOnSelect expand='sm' bg='light' variant='light'>
      <Container className={classes.container}>
        <Navbar.Brand>Nanobody Polyreactivity</Navbar.Brand>
        <Navbar.Toggle aria-controls='responsive-navbar-nav' />
        <Navbar.Collapse id='responsive-navbar-nav'>
          <Nav className='me-auto'>
            <Nav.Link as={Link} exact='true' to='/'>Home</Nav.Link>
            <Nav.Link as={Link} exact='true' to='/results'>Results</Nav.Link>
            <Nav.Link as={Link} exact='true' to='/faq'>FAQ</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  )
}
