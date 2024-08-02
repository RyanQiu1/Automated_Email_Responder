import React, { useEffect, useState } from "react";
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Menu,
  MenuItem,
} from "@mui/material";
import EmailDetails from "./EmailDetails";
import { useDataContext } from "./DataContext";


interface EditDraftUIProps {
  onSave: (content: string) => void;
  initialContent?: string;
  email: {
    id: number;
    subject: string;
    sender: string;
    receivedAt: string;
    body: string;
  };
}
const categories: string[] = [
  "Blackmail",
  "Brute Force Attack",
  "Business Email Compromised",
  "Case Enquiry",
  "Compromised",
  "Cyber Crime",
  "Cyber Crime - Reported to police",
  "Defacement",
  "Email Notification of Data Breaches",
  "Extortion Email",
  "Fake, Possible Scam or Impersonation Website",
  "Harassment",
  "Insufficient Information",
  "Malware Hosting App",
  "Non-Cybersecurity Related Reports",
  "Phishing",
  "Possible Fake/Impersonation Sites",
  "Ransomware",
  "Scam/Gambling/Investment/Unlicensed Money Lending/ Pornography Sites",
  "Social Media Impersonation",
  "Spam Email",
  "Spoofed Email",
  "Spyware",
  "Tech Support Scam",
  "Vishing",
];

const EditDraftUI: React.FC<EditDraftUIProps> = ({
  onSave,
  initialContent = "",
  email,
}) => {
  const [content, setContent] = useState(initialContent);
  const [successMessage, setSuccessMessage] = useState(""); // [1]
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);// [2]
  const [predictionEmailId, setPredictionEmailId] = useState<number>(1); // Start with email ID 1
  const maxEmailId = 2; 

  const handleSave = () => {
    onSave(content);
    // Display a success message
    setSuccessMessage("Draft saved successfully!");
    // Clear the success message after 3 seconds

    setTimeout(() => {
      setSuccessMessage("");
    }, 3000);
  };

  // Backbutton



  const handleClick = async (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/retrain_model",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Label": event.currentTarget.innerText,
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

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleSelect = (category: string) => {
    // Send to backend

    setSuccessMessage(
      "Feedback sent successfully! Model will be trained and updated!"
    );
    setTimeout(() => {
      setSuccessMessage("");
    }, 3000);

    handleClose(); // Close the dropdown after selection
    console.log("Selected Category:", category); // Here you can also handle sending feedback or other actions
  };

  const handleAnotherPrediction = async () => {
    try {

      setPredictionEmailId(prevId => prevId >= maxEmailId ? 0 : prevId + 1);
      // Assuming `email.id` is defined and contains the index of the email
      const url = `http://127.0.0.1:5000/get_another_prediction?prediction_email_index=${predictionEmailId}`;
      const response = await fetch(url, {
        method: "GET", // Changed to GET method
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const result = await response.json();
        setContent(result.second_template);
        setCat(result.second_category);
        console.log("Prediction retrieved successfully:", result);
        // Handle success (e.g., show a success message or process the result)
      } else {
        console.error("Failed to retrieve prediction:", response.statusText);
        // Handle failure (e.g., show an error message)
      }
    } catch (error) {
      console.error("Error during prediction retrieval:", error);
      // Handle error (e.g., show an error message)
    }
  };

  const { fetchedData } = useDataContext();
  const [cat, setCat] = useState("");
  const [subCat, setSubCat] = useState("");

  useEffect(() => {
    if (fetchedData) {
      const JsonData = JSON.parse(fetchedData);
      setCat(JsonData.category);
      setSubCat(JsonData.sub_category || ""); // Handle case where sub_category might be undefined
      setContent(JsonData.template);
    }
  }, [fetchedData]);


  return (
    <Box sx={{ display: "flex", gap: "20px" }}>
      <Box sx={{ flex: 1 }}>
        <EmailDetails
          showSaveButton={false}
          email={email}
          setTabValue={function (value: React.SetStateAction<number>): void {
            throw new Error("Function not implemented.");
          }}
        />
      </Box>
      <Paper sx={{ flex: 1, padding: "20px" }}>
        <Typography variant="subtitle1">
          Predicted Category of Email: <strong>{cat + " " + subCat}</strong>
        </Typography>
        <Typography variant="subtitle1">
          <i>
            <br />
            Edit your Email Draft here:
          </i>
        </Typography>
        <TextField
          multiline
          fullWidth
          rows={10}
          variant="outlined"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          sx={{ marginTop: "20px" }}
        />
        <Button
          variant="contained"
          onClick={handleSave}
          sx={{ marginTop: "20px", color: "black", backgroundColor: "#90caf9" }}
        >
          Save Draft
        </Button>
        <Button
          variant="outlined"
          onClick={handleAnotherPrediction}
          sx={{ marginLeft: "30px", marginTop: "20px", color: "black" }}
        >
          Get Another Prediction
        </Button>
        <Button
          variant="outlined"
          onClick={handleClick}
          sx={{ marginLeft: "30px", marginTop: "20px", color: "black" }}
        >
          Feedback
        </Button>
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleClose}
        >
          {categories.map((category) => (
            <MenuItem key={category} onClick={() => handleSelect(category)}>
              {category}
            </MenuItem>
          ))}
        </Menu>

        <Typography sx={{ marginTop: "20px", color: "green" }}>
          {successMessage}
        </Typography>
      </Paper>
    </Box>
  );
};

export default EditDraftUI;
