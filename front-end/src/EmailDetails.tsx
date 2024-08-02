import React, { useState } from "react";
import {
  Paper,
  Typography,
  Box,
  Button,
  CircularProgress,
} from "@mui/material";
import { useDataContext } from "./DataContext";

interface Email {
  id: number;
  subject: string;
  sender: string;
  receivedAt: string;
  body: string;
}

interface EmailDetailsProps {
  email: Email;
  setTabValue?: React.Dispatch<React.SetStateAction<number>>;
  showSaveButton?: boolean;
}



const EmailDetails: React.FC<EmailDetailsProps> = ({
  email,
  setTabValue,
  showSaveButton,
}) => {
  const [loading, setLoading] = useState(false);
  const { setFetchedData } = useDataContext();
  console.log(email.id);


  const handleCreateDraft = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:5000/load_model_and_predict?email_index=" + email.id
      );
      const data = await response.json();
      setFetchedData(data); // Set the fetched data in state
      if (setTabValue) {
        setTabValue(2); // Only call setTabValue if it's defined
      }
    } catch (error) {
      console.error("Error fetching email data:", error);
    }
    setLoading(false);
  };



  return (
    <Box sx={{ mx: "auto", mt: 0 }}>
      {/* Subject Section */}
      <Paper sx={{ padding: "20px", marginBottom: "20px", display: "flex", alignItems: "center"}}>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
          Subject:
        </Typography>
        <Typography variant="body1" sx={{ marginLeft: "5px"}}>{email.subject}</Typography>
      </Paper>

      {/* From Section */}
      <Paper sx={{ padding: "20px", marginBottom: "20px", display: "flex", alignItems: "center"}}>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
          From:
        </Typography>
        <Typography variant="body1" sx={{ marginLeft: "5px"}}>{email.sender}</Typography>
      </Paper>

      {/* Received At Section */}
      <Paper sx={{ padding: "20px", marginBottom: "20px", display: "flex", alignItems: "center"}}>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
          Received At:
        </Typography>
        <Typography variant="body1" sx={{ marginLeft: "5px"}}>{email.receivedAt}</Typography>
      </Paper>

      {/* Body Section */}
      <Paper sx={{ padding: "20px", marginBottom: "20px" }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
          Body: <br/>
        </Typography>
        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
          {email.body}
        </Typography>
      </Paper>

      {/* Actions Section */}
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {showSaveButton && (
          <Button
            variant="contained"
            onClick={handleCreateDraft}
            sx={{
              backgroundColor: "#90caf9",
              color: "black",
            }}
          >
            Create Draft
          </Button>
        )}
        {loading && <CircularProgress sx={{ marginLeft: '20px' }} />}
      </Box>
    </Box>
  );
};

export default EmailDetails;
