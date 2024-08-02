import React, { useState, useEffect } from "react";
import {
  TextField,
  Button,
  Typography,
  Paper,
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { useDataContext } from "./DataContext";

const SettingPage = () => {
  const [name, setName] = useState<string>(""); // Initialize with the current name, if available
  const [email, setEmail] = useState<string>(""); // Initialize with the current email, if available
  const [emails, setEmails] = useState<string[]>([]); // List of fetched email addresses
  const [isSaved, setIsSaved] = useState<boolean>(false);
  const [isEmailSaved, setIsEmailSaved] = useState<boolean>(false);

  const { setEmailAddress } = useDataContext();

  useEffect(() => {
    fetchEmailsAddressAvailable();
  }, []);

  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
    setIsSaved(false); // Reset saved status on change
  };

  const handleSaveName = async () => {
    console.log("Name:", name);
    try {
      const response = await fetch("http://127.0.0.1:5000/save_designation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify({ name: name }),
      });
      const data = await response.json();
      console.log("Response data:", data);
      setIsSaved(true);
    } catch (error) {
      console.error("Error during POST request:", error);
    }
  };

  const fetchEmailsAddressAvailable = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/get_email_accounts", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
      const data = await response.json();

      // Assuming the response is an array of strings (email addresses)
      if (Array.isArray(data)) {
        setEmails(data);
      } else {
        console.error("Unexpected response format:", data);
      }
    } catch (error) {
      console.error("Error during GET request:", error);
    }
  };

  const handleEmailChange = (event: SelectChangeEvent<string>) => {
    setEmail(event.target.value);
    setIsEmailSaved(false);
  };

  const handleSaveEmail = async () => {
    // Sent Post request to save email
    try {
      await fetch("http://127.0.0.1:5000/save_email_in_db", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify({ email: email }),
      });
      setEmailAddress(email);
      setIsEmailSaved(true);
    } catch (error) {
      console.error("Error during POST request:", error);
    }
  };

  return (
    <>
      <Paper
        elevation={1}
        sx={{ marginBottom: "20px", p: 3, maxWidth: "2000px" }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
          Setup/Settings
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 0 }}>
          You will be directed to set your name and email address to get started
          with the email responder. You can also manually change your name and
          email address here.
        </Typography>
      </Paper>
      <Paper elevation={3} sx={{ padding: 3, margin: 2, mx: "auto" }}>
        
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 2 }}>
          Change your mail designation/name:
        </Typography>
        <TextField
          fullWidth
          label="Name"
          variant="outlined"
          value={name}
          onChange={handleNameChange}
          sx={{ mb: 2 }}
        />
        <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
          <Button variant="contained" color="primary" onClick={handleSaveName}>
            Save Name
          </Button>
        </Box>
        {isSaved && (
          <Typography color="primary" sx={{ mt: 2 }}>
            Name saved successfully!
          </Typography>
        )}

        <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 2 }}>
          Change your Email:
        </Typography>
        <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
          <InputLabel>Email</InputLabel>
          <Select value={email} onChange={handleEmailChange} label="Email">
            {emails.map((email, index) => (
              <MenuItem key={index} value={email}>
                {email}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
          <Button variant="contained" color="primary" onClick={handleSaveEmail}>
            Save Email
          </Button>
        </Box>
        {isEmailSaved && (
          <Typography color="primary" sx={{ mt: 2 }}>
            Email saved successfully!
          </Typography>
        )}

        <Typography variant="subtitle1" sx={{ mt: 2 }}>
          <strong>Note:</strong> Please Select your email address from the
          dropdown list.
        </Typography>
      </Paper>
    </>
  );
};

export default SettingPage;
