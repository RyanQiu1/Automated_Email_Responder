import React from 'react';
import { Container, Typography, Box } from '@mui/material';
const NotFoundPage: React.FC = () => {


  return (
    <Container maxWidth="sm">
      <Box sx={{ textAlign: 'center', marginTop: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          404: The page you are looking for isn’t here
        </Typography>
        <Typography variant="subtitle1">
          You might have the wrong address, or the page may have moved.
        </Typography>
      </Box>
    </Container>
  );
};

export default NotFoundPage;
