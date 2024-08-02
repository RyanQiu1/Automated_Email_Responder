import React, { useState, useEffect } from "react";
import {
  Button,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Paper,
  Typography,
  CircularProgress,
  Tooltip,
  IconButton,
  Divider,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";

const ChangeModelPage: React.FC = () => {
  // A static list of models if needed or default models
  const staticModels: string[] = ["Local_LLM"];

  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // API call to fetch models
  useEffect(() => {
    fetch("http://localhost:5000/get_models")
      .then((response) => response.json())
      .then((data) => {
        // Combine fetched models with static models if fetched list is empty
        setModels([...data.models, ...staticModels]);
        setLoading(false);
      })
      .catch((error: Error) => {
        console.error("Failed to fetch models:", error);
        setModels(staticModels); // Use static models if fetch fails
        setError("Failed to load models.");
        setLoading(false);
      });
  }, []);

  // Command to send api request to change model
  const handleChangeModel = (modelName: string) => {
    fetch(`http://localhost:5000/change_model`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_name: modelName }), // Assuming the backend expects a JSON body with model_name key
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to change model.");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Model changed successfully:", data);
      })
      .catch((error: Error) => {
        console.error("Failed to change model:", error);
      });
  };

  const handleClickOpen = (modelName: string) => {
    setSelectedModel(modelName);
    handleChangeModel(modelName); // Call the change model API
    setOpenDialog(true);
  };

  const handleClose = () => {
    setOpenDialog(false);
  };

  const handleConfirmChange = () => {
    if (selectedModel) {
      console.log("Switching to model:", selectedModel);
      alert(`Model changed to ${selectedModel}`);
      setOpenDialog(false);
    }
  };

  return (
    <>
      <Paper
        elevation={1}
        sx={{ marginBottom: "20px", p: 3, maxWidth: "2000px" }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
          Change Models
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 0 }}>
          Click on a model to change the current model used for processing.

          <Tooltip
            title="This models are available for to change, currently only changing the Local_LLM works, the rest is still a work in progress."
            arrow
          >
            <IconButton sx={{ ml: 0 }}>
              <InfoIcon />
            </IconButton>
          </Tooltip>
         If green, it is model that is available to change.
        </Typography>
      </Paper>

      <Paper elevation={1} sx={{ mt: 1, p: 3 }}>
        <Typography
          variant="h6"
          component="h2"
          sx={{ fontWeight: "bold", mb: 2 }}
        >
          Models Available for Change 
        </Typography>
        {error && <Typography color="error">{error}</Typography>}
        {loading ? (
          <CircularProgress />
        ) : (
          <List>
            {models.length > 0 ? (
              models.map((modelName, index) => (
                <React.Fragment key={index}>
                  <ListItem
                    button
                    onClick={() => handleClickOpen(modelName)}
                    sx={{
                      backgroundColor:
                        (modelName === "Local_LLM" || modelName === "final_ensemble_model.pkl") ? "lightgreen" : "inherit",
                    }}
                  >
                    <ListItemText
                      sx={{
                        color: modelName === "Local_LLM" || modelName === "final_ensemble_model.pkl" ? "green" : "black",
                        fontWeight:
                          modelName === "SpecialModel" ? "bold" : "normal",
                      }}
                      primary={`${index + 1}. ${modelName}`}
                    />
                  </ListItem>
                  {index < models.length - 1 && <Divider />}
                </React.Fragment>
              ))
            ) : (
              <ListItem>
                <ListItemText primary="No models available" />
              </ListItem>
            )}
          </List>
        )}
      </Paper>
      <Dialog open={openDialog} onClose={handleClose}>
        <DialogTitle>Confirm Model Change</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to change to the model "{selectedModel}"?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button onClick={handleConfirmChange} autoFocus>
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ChangeModelPage;
