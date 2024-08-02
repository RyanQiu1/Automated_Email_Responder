import React, { useState } from 'react';
import { Button, TextField, Paper, Typography, Box } from '@mui/material';

interface Email {
  id: number;
  subject: string;
  receivedAt: string;
  sender: string;
  body: string;
}

interface EmailEditorProps {
  email: Email;
  onSave: (updatedEmail: Email) => void;
}



const EmailEditor: React.FC<EmailEditorProps> = ({ email, onSave }) => {
  const [editedEmail, setEditedEmail] = useState<Email>(email);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleBodyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEditedEmail({ ...editedEmail, body: event.target.value });
  };

  const handleSave = () => {
    handleSaveDraft(editedEmail.body);
  };

  const handleSaveDraft = async (content: string) => {
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/save_existing_draft_email?email_index=${email.id}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ content }),
        }
      );

      if (response.ok) {
        const result = await response.json();
        console.log("Draft saved successfully:", result);
        onSave(editedEmail);
        setSuccessMessage("Draft saved successfully!");
      } else {
        console.error("Failed to save draft:", response.statusText);
        // Handle failure (e.g., show an error message)
      }
    } catch (error) {
      console.error("Error saving draft:", error);
      // Handle error (e.g., show an error message)
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2, margin: 'auto', mt: 2}}>
      <Typography variant="h6" gutterBottom>
        Editing: {email.subject}
      </Typography>
      <Typography variant="subtitle1" gutterBottom>
        From: {email.sender} at {email.receivedAt}
      </Typography>
      <TextField
        label="Email Body"
        multiline
        fullWidth
        variant="outlined"
        value={editedEmail.body}
        onChange={handleBodyChange}
        sx={{ mb: 2 }}
      />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="contained" color="primary" onClick={handleSave}>
          Save Changes
        </Button>
      </Box>
        {successMessage && (
            <Typography sx={{ mt: 2, color: 'success.main' }}>{successMessage}</Typography>
        )}
    </Paper>
  );
};

export default EmailEditor;
