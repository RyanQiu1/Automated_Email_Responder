import React, { useState } from 'react';
import { Container, Paper, Typography, Button, Grid, Box } from '@mui/material';

// Define types for clarity
interface DataItem {
  id: number;
  feature1: string;
  feature2: number;
}

interface Prediction {
  id: number;
  predictionResult: string;
}

const PredictedPage: React.FC = () => {
  // Dummy data for display
  const [dataItems, setDataItems] = useState<DataItem[]>([
    { id: 1, feature1: 'Sample A', feature2: 200 },
    { id: 2, feature1: 'Sample B', feature2: 150 },
    { id: 3, feature1: 'Sample C', feature2: 300 },
  ]);

  const [predictions, setPredictions] = useState<Prediction[]>([]);

  const handlePredict = () => {
    // Simulate predictions based on dataItems
    const predictedResults: Prediction[] = dataItems.map(item => ({
      id: item.id,
      predictionResult: `Result for ${item.feature1} is ${item.feature2 > 250 ? 'High' : 'Low'}`
    }));

    setPredictions(predictedResults);
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ padding: 3, marginTop: 4 }}>
        <Typography variant="h5" component="h2" sx={{ marginBottom: 2 }}>
          Prediction Page
        </Typography>
        <Button variant="contained" color="primary" onClick={handlePredict}>
          Predict
        </Button>
        <Box sx={{ marginY: 2 }}>
          <Typography variant="h6">Input Data:</Typography>
          <Grid container spacing={2}>
            {dataItems.map(item => (
              <Grid item xs={12} md={6} key={item.id}>
                <Paper sx={{ padding: 2 }}>
                  <Typography>Feature 1: {item.feature1}</Typography>
                  <Typography>Feature 2: {item.feature2}</Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>
        {predictions.length > 0 && (
          <Box sx={{ marginTop: 2 }}>
            <Typography variant="h6">Predicted Results:</Typography>
            {predictions.map(prediction => (
              <Typography key={prediction.id}>
                {prediction.predictionResult}
              </Typography>
            ))}
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default PredictedPage;
