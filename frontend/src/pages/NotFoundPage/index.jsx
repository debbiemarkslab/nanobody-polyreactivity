import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';

export default function NotFoundPage() {
  return (
    <Container style={{ paddingTop: 40 }}>
      <Row className="justify-content-md-center align-items-center">
        <h4>Page Not Found</h4>
      </Row>
    </Container>
  );
}
