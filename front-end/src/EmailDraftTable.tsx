import React, { useState, useEffect } from 'react';
import { CircularProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, IconButton, Box, Tooltip } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import EmailEditor from './EmailEditor'; // Assuming EmailEditor is correctly imported
import { useDataContext } from './DataContext';
import InfoIcon from '@mui/icons-material/Info';

interface Email {
  id: number;
  subject: string;
  receivedAt: string;
  sender: string;
  body: string;
}

interface EmailState {
  draftEmails: Email[];
  loading: boolean;
  error: string;
}

const EmailDraftTable: React.FC = () => {
  const [emailState, setEmailState] = useState<EmailState>({ draftEmails: [], loading: false, error: '' });
  const [selectedEmailId, setSelectedEmailId] = useState<number | null>(null);
  const [email, setEmail] = useState<Email | null>(null);

  const handleEmailClick = (emailId: number) => {
    setSelectedEmailId(emailId);
    const foundEmail = emailState.draftEmails.find(email => email.id === emailId);
    if (foundEmail) {
      setEmail(foundEmail);
    }
  };

  const handleBackClick = () => {
    setSelectedEmailId(null);
  };

  const { emailaddress } = useDataContext(); // Assuming useDataContext is correctly imported

  useEffect(() => {
    // Only fetch if draftEmails is empty
    if (emailState.draftEmails.length === 0) {
      setEmailState(prevState => ({ ...prevState, loading: true }));
      fetch(`http://localhost:5000/get_draft_emails?email=${emailaddress}`)
        .then(response => {
          if (!response.ok) throw new Error('Network response was not ok');
          return response.json();
        })
        .then(data => {
          setEmailState({ draftEmails: data, loading: false, error: '' });
        })
        .catch(error => {
          console.error('Failed to fetch draft emails:', error);
          setEmailState({ draftEmails: [], loading: false, error: 'Failed to load draft emails' });
        });
    }
  }, [emailaddress, emailState.draftEmails.length]); // dependency on emailaddress

  if (emailState.error) return <Typography color="error">{emailState.error}</Typography>;
  if (emailState.loading) {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "100%",
        }}
      >
        <CircularProgress />
        <Typography variant="subtitle1" sx={{ marginTop: "20px" }}>
          Loading...
        </Typography>
      </Box>
    );
  }
  if (selectedEmailId !== null && email) {
    return (
      <>
        <IconButton onClick={handleBackClick} color="inherit">
          <ArrowBackIcon />
        </IconButton>
        <EmailEditor email={email} onSave={(updatedEmail) => setEmail(updatedEmail)} />
      </>
    );
  }

  return (
    <>
    <Paper
        elevation={1}
        sx={{ marginBottom: "20px", p: 3, maxWidth: "2000px" }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
          Draft Emails
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 0 }}>
          Click on the draft to view or edit the draft.
          <Tooltip title="These are draft emails in your draft inbox." arrow>
            <IconButton sx={{ ml: 0 }}>
              <InfoIcon />
            </IconButton>
          </Tooltip>
        </Typography>
      </Paper>
    <TableContainer component={Paper}>
      <Table aria-label="Draft emails table">
        <TableHead>
          <TableRow>
            <TableCell align="left" sx={{ fontWeight: "bold", fontSize: "20px" }}>Subject</TableCell>
            <TableCell align="right" sx={{ fontWeight: "bold", fontSize: "20px" }}>Date</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {emailState.draftEmails.map((email) => (
            <TableRow key={email.id} onClick={() => handleEmailClick(email.id)} hover>
              <TableCell align="left" sx={{ fontSize: "15px" }}>{email.subject}</TableCell>
              <TableCell align="right" sx={{ fontSize: "15px" }}>{email.receivedAt}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
    </>
  );
}

export default EmailDraftTable;
