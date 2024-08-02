import React, { useState, useEffect } from "react";
import {
  Box,
  AppBar,
  Typography,
  IconButton,
  Paper,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  CircularProgress,
  Drawer,
  ListItemIcon,
  ListItem,
  ListItemText,
  List,
  Toolbar,
  SvgIcon,
  Tooltip,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import EditDraftUI from "./EditDraftUI";
import EmailDetails from "./EmailDetails";
import MenuIcon from "@mui/icons-material/Menu";
import MailIcon from "@mui/icons-material/Mail";
import EditIcon from "@mui/icons-material/Edit";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import SettingsIcon from "@mui/icons-material/Settings";
import EmailDraftTable from "./EmailDraftTable";
import { useDataContext } from "./DataContext";
import SettingPage from "./SettingPage";
import { ReactComponent as SingCERTicon } from "./Images/Singcerticon.svg";
import BatchPredictionIcon from "@mui/icons-material/BatchPrediction";
import ChangeModelPage from "./ChangeModelPage";
import NotFoundPage from "./NotFound";
import SummarizeIcon from "@mui/icons-material/Summarize";
import SummaryPage from "./SummaryPage";
import EmailTemplate from "./EmailTemplate";
import AllInboxIcon from "@mui/icons-material/AllInbox";
import FetchedPage from "./FetchedPage";
import InfoIcon from "@mui/icons-material/Info";
import SendTwoToneIcon from "@mui/icons-material/SendTwoTone";

// Define the Email interface
interface Email {
  id: number;
  subject: string;
  sender: string;
  receivedAt: string;
  body: string;
}

// Create a theme
const theme = createTheme({
  palette: {
    background: {
      default: "#FFFFFF", // Set the default background color to white
    },
    text: {
      primary: "#FFFFF", // Light blue text color
    },
  },
  typography: {
    fontFamily: "Arial, sans-serif", // Set the font family to Roboto, sans-serif
    fontSize: 14,
  },
});

const EmailTable: React.FC<{
  onEmailClick: (id: number) => void;
  emails: Email[] | null; // Allow emails to be null
}> = ({ onEmailClick, emails }) => {
  // Handle the case where emails might be null
  if (emails === null) {
    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: "bold", fontSize: "20px" }}>
                Subject
              </TableCell>
              <TableCell
                align="left"
                sx={{ fontWeight: "bold", fontSize: "20px" }}
              >
                Sender
              </TableCell>
              <TableCell
                align="right"
                sx={{ fontWeight: "bold", fontSize: "20px" }}
              >
                Received At
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell colSpan={3} align="center">
                No emails available
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: "bold", fontSize: "20px" }}>
              Subject
            </TableCell>
            <TableCell
              align="left"
              sx={{ fontWeight: "bold", fontSize: "20px" }}
            >
              Sender
            </TableCell>
            <TableCell
              align="right"
              sx={{ fontWeight: "bold", fontSize: "20px" }}
            >
              Received At
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {emails.map((email: Email) => (
            <TableRow
              key={email.id}
              onClick={() => onEmailClick(email.id)}
              style={{ cursor: "pointer" }}
            >
              <TableCell component="th" scope="row" sx={{ fontSize: "15px" }}>
                {email.subject}
              </TableCell>
              <TableCell align="left" sx={{ fontSize: "15px" }}>
                {email.sender}
              </TableCell>
              <TableCell align="right" sx={{ fontSize: "15px" }}>
                {email.receivedAt}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const ChatBotUI: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedEmailId, setSelectedEmailId] = useState<number | null>(null);
  const [emails, setEmails] = useState<Email[]>([]);
  const [loading, setLoading] = useState(true);
  const [drawerOpen, setDrawerOpen] = React.useState(true); // Default to closed
  const { emailaddress } = useDataContext();
  const { setEmailAddress } = useDataContext();
  const [selectedIndex, setSelectedIndex] = useState(null);



  const handleSaveDraft = async (content: string) => {
    try {
      let new_index = 0;
      if (selectedEmailId !== null) {
        new_index = emails.length - selectedEmailId;
      }
      const response = await fetch(
        `http://127.0.0.1:5000/save_draft_response?email_index=${new_index}`,
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
        // Handle success (e.g., show a success message)
      } else {
        console.error("Failed to save draft:", response.statusText);
        // Handle failure (e.g., show an error message)
      }
    } catch (error) {
      console.error("Error saving draft:", error);
      // Handle error (e.g., show an error message)
    }
  };

  useEffect(() => {
    const fetchEmailAddressAndEmails = async () => {
      try {
        // Fetch email address
        const emailResponse = await fetch(
          "http://127.0.0.1:5000/check_email_address"
        );
        const emailData = await emailResponse.json();
        const fetchedEmailAddress = emailData["email"];

        // If an email address is found, fetch unread emails
        if (fetchedEmailAddress !== "" || emailaddress !== "") {
          const unreadEmailsResponse = await fetch(
            `http://127.0.0.1:5000/get_unread_emails?email=${fetchedEmailAddress}`
          );
          const unreadEmailsData = await unreadEmailsResponse.json();
          setEmailAddress(fetchedEmailAddress);
          // Convert string to array if necessary
          const parsedData = JSON.parse(unreadEmailsData);
          setLoading(false);
          setEmails(parsedData);
          setTabValue(0);
        } else {
          setLoading(false);
          setTabValue(3);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchEmailAddressAndEmails();
  }, [emailaddress, setEmailAddress]); // Ensure that `emailaddress` is properly defined and managed

  useEffect(() => {
    const fetchUnreadEmails = async () => {
      try {
        // Check if emailaddress is set and not empty
        if (emailaddress !== "") {
          // Fetch unread emails
          const unreadEmailsResponse = await fetch(
            `http://127.0.0.1:5000/get_unread_emails?email=${emailaddress}`
          );
          const unreadEmailsData = await unreadEmailsResponse.json();

          // Convert string to array if necessary
          const parsedData = JSON.parse(unreadEmailsData);
          setEmails(parsedData);
          setLoading(false);
          setTabValue(0);
        } else {
          setLoading(false);
          setTabValue(3); // Set tab to 3 if email address is empty
        }
      } catch (error) {
        console.error("Error fetching unread emails:", error);
        setLoading(false);
      }
    };

    fetchUnreadEmails();
  }, [emailaddress]); // Dependency on emailaddress

  const handleEmailClick = (emailId: number) => {
    setSelectedEmailId(emailId);
  };

  const handleBackClick = () => {
    setSelectedEmailId(null);
    setTabValue(0); // Switch back to "View Latest Emails" tab
  };

  let { fetchedData } = useDataContext();

  const handleBackClickSelected = () => {
    setSelectedEmailId(null);
    fetchedData = false;
    setTabValue(0);
  };

  const handleListItemClick = (index) => {
    setSelectedIndex(index);
    setTabValue(index); // Your existing function to handle tab change
  };

  const appBarHeight = "90px";
  const drawerWidth = 240;

  useEffect(() => {
    console.log("Loading:", loading);
    if (!loading) {
      window.scrollTo(0, 0); // Scroll to the top when loading is complete
    }
  }, [loading]);

  useEffect(() => {}, [tabValue]);

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          display: "flex",
          height: "100vh",
          backgroundColor: theme.palette.background.paper,
        }}
      >
        <AppBar
          position="fixed"
          sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
        >
          <Box
            sx={{
              display: "flex",
              height: appBarHeight,
              justifyContent: "space-between",
            }}
          >
            <Toolbar>
              <IconButton
                edge="start"
                color="inherit"
                aria-label="menu"
                onClick={() => setDrawerOpen(!drawerOpen)}
              >
                <MenuIcon />
              </IconButton>
              <Typography
                variant="h5"
                component="div"
                sx={{ flexGrow: 1, ml: 2 }}
              >
                SingCERT AlphaReply
              </Typography>
              <SendTwoToneIcon sx={{ fontSize: 30, ml: 2.5 }} />
            </Toolbar>
            <SvgIcon
              component={SingCERTicon}
              sx={{ fontSize: 200, marginLeft: "20px", alignItems: "right" }}
              viewBox="0 0 600 476.6"
            ></SvgIcon>
          </Box>
        </AppBar>
        <Drawer
          variant="persistent"
          open={drawerOpen}
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: drawerWidth,
              boxSizing: "border-box",
              top: appBarHeight,
            },
          }}
        >
          <List>
            {[
              "View Latest Emails",
              "Edit Draft",
              "Predict Category",
              "Settings",
              "Change Model",
              "Summarise Email",
              "Email Templates",
            ].map((text, index) => (
              <ListItem
                button
                key={text}
                onClick={() => handleListItemClick(index)}
                sx={{
                  backgroundColor:
                    selectedIndex === index
                      ? "rgba(102, 153, 255, 0.2)"
                      : "transparent", // Light blue background if selected
                  "&:hover": {
                    backgroundColor: "rgba(102, 153, 255, 0.2)", // Slightly darker blue on hover
                  },
                }}
              >
                <ListItemIcon>
                  {index === 0 && <MailIcon />}
                  {index === 1 && <EditIcon />}
                  {index === 2 && <BatchPredictionIcon />}
                  {index === 3 && <SettingsIcon />}
                  {index === 4 && <ModelTrainingIcon />}
                  {index === 5 && <SummarizeIcon />}
                  {index === 6 && <AllInboxIcon />}
                </ListItemIcon>
                <ListItemText primary={text} />
              </ListItem>
            ))}
          </List>
        </Drawer>
        <Box
          sx={{
            padding: "20px",
            flexGrow: 1,
            p: 3,
            mt: appBarHeight,
            transition: theme.transitions.create("margin", {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
            marginLeft: drawerOpen ? "0px" : "-240px",
            marginBottom: "20px",
            overflow: "auto", // Allows scrolling within the box
          }}
        >
          {tabValue === 0 && (
            <>
              {selectedEmailId ? (
                <>
                  <IconButton onClick={handleBackClick} color="inherit">
                    <ArrowBackIcon sx={{ color: "black" }} />
                  </IconButton>
                  <EmailDetails
                    email={
                      emails.find((email) => email.id === selectedEmailId)!
                    }
                    showSaveButton={true}
                    setTabValue={setTabValue}
                  />
                </>
              ) : loading ? (
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
              ) : (
                <>
                  <Paper
                    elevation={1}
                    sx={{ marginBottom: "20px", p: 3, maxWidth: "2000px" }}
                  >
                    <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
                      View Latest Unread Emails
                    </Typography>
                    <Typography variant="subtitle1" sx={{ mb: 0 }}>
                      Click on an email to view it and predict its category. It
                      will also retrieve the email template upon completion.
                      <Tooltip
                        title="This are the latest unread emails available in the email you have set, do click on them to view the email content and predict the category."
                        arrow
                      >
                        <IconButton sx={{ ml: 0 }}>
                          <InfoIcon />
                        </IconButton>
                      </Tooltip>
                    </Typography>
                  </Paper>

                  <EmailTable onEmailClick={handleEmailClick} emails={emails} />
                </>
              )}
            </>
          )}
          {tabValue === 1 && <EmailDraftTable></EmailDraftTable>}
          {tabValue === 2 && fetchedData && selectedEmailId && (
            <>
              <IconButton onClick={handleBackClickSelected} color="inherit">
                <ArrowBackIcon sx={{ color: "black" }} />
              </IconButton>
              <EditDraftUI
                onSave={handleSaveDraft}
                initialContent=""
                email={emails.find((email) => email.id === selectedEmailId)!}
              />
            </>
          )}
          {tabValue === 2 &&
            (fetchedData ? (
              <FetchedPage></FetchedPage>
            ) : (
              <NotFoundPage></NotFoundPage>
            ))}
          {tabValue === 3 && <SettingPage></SettingPage>}
          {tabValue === 4 && <ChangeModelPage></ChangeModelPage>}
          {tabValue === 5 && <SummaryPage emails={emails} />}
          {tabValue === 6 && <EmailTemplate></EmailTemplate>}
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default ChatBotUI;
