import React, { useState, useEffect } from 'react';
import { useDataContext } from "./DataContext";
import { Typography, Paper, Box } from '@mui/material';

const FetchedPage: React.FC = () => {
  const { fetchedData } = useDataContext();

  const [cat, setCat] = useState<string>("");
  const [subCat, setSubCat] = useState<string>("");
  const [content, setContent] = useState<string>("");

  useEffect(() => {
    if (fetchedData) {
      try {
        const jsonData = JSON.parse(fetchedData);
        setCat(jsonData.category);
        setSubCat(jsonData.sub_category || "");
        setContent(jsonData.template);
      } catch (error) {
        console.error("Failed to parse fetched data:", error);
      }
    }
  }, [fetchedData]);

  return (
    <Box
      sx={{
        maxWidth: 800,
        mx: 'auto',
        mt: 4,
        p: 3,
        bgcolor: 'background.paper',
        boxShadow: 3,
        borderRadius: 2,
      }}
    >
      <Paper elevation={3} sx={{ padding: 2, mb: 3 }}>
        <Box sx={{ mb: 3, p: 2, border: '1px solid #ccc', borderRadius: 1 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Category
          </Typography>
          <Typography variant="body1">
            {cat}
          </Typography>
        </Box>
        
        <Box sx={{ mb: 3, p: 2, border: '1px solid #ccc', borderRadius: 1 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Sub-Category
          </Typography>
          <Typography variant="body1">
            {subCat}
          </Typography>
        </Box>
        
        <Box sx={{ p: 2, border: '1px solid #ccc', borderRadius: 1 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Content
          </Typography>
          <Typography variant="body1">
            {content}
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default FetchedPage;
