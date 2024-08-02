import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Paper,
  Button,
  Typography,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TableContainer,
  IconButton,
  Tooltip,
  Box,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import AddBoxTwoToneIcon from '@mui/icons-material/AddBoxTwoTone';

type Template = {
  subject: string;
  body: string;
};

const EmailTemplates: React.FC = () => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(
    null
  );
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [form, setForm] = useState({ subject: "", body: "" });

  useEffect(() => {
    axios
      .get("http://127.0.0.1:5000/get_email_templates")
      .then((response) => setTemplates(response.data.email_templates))
      .catch((error) => console.error("Error fetching templates:", error));
  }, [setTemplates]);

  const handleOpenEdit = (template: Template) => {
    setSelectedTemplate(template);
    setForm(template);
    setIsEditOpen(true);
  };

  const handleSaveEdit = () => {
    if (!selectedTemplate) return;
    axios
      .post("http://127.0.0.1:5000/update_email_template", form)
      .then(() => {
        setTemplates((prev) =>
          prev.map((t) =>
            t.subject === selectedTemplate.subject ? { ...t, ...form } : t
          )
        );
        setIsEditOpen(false);
      })
      .catch((error) => console.error("Error updating template:", error));
  };

  const handleOpenAdd = () => {
    setForm({ subject: "", body: "" });
    setIsAddOpen(true);
  };

  const handleAddTemplate = () => {
    axios
      .post("http://127.0.0.1:5000/add_email_template", form)
      .then((response) => {
        setTemplates((prev) => [...prev, response.data]);
        setIsAddOpen(false);
      })
      .catch((error) => console.error("Error adding template:", error));
  };

  

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <>
      <Paper
        elevation={1}
        sx={{ marginBottom: "20px", p: 3, maxWidth: "2000px" }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
          Email Templates
        </Typography>
        <Box sx={{ display: "flex", alignContent: "center", justifyItems: "center" }}>
          <Typography variant="subtitle1" sx={{ mb: 0 }}>
            Click on an email template to view below, or click the add button
            icon on the left to add a new template.
            <Tooltip
              title="This are the email templates available in the database, do add/update them if needed."
              arrow
            >
              <IconButton sx={{ ml: 0 }}>
                <InfoIcon />
              </IconButton>
            </Tooltip>
          </Typography>
          <IconButton
            onClick={handleOpenAdd}
  
            color="primary"
          >
            <AddBoxTwoToneIcon fontSize="medium"/>
          </IconButton>
        </Box>
      </Paper>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: "bold", fontSize: "20px" }}>Subject</TableCell>
              <TableCell sx={{ fontWeight: "bold", fontSize: "20px" }}>Edit</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {templates.map((template, index) => (
              <TableRow key={index} >
                <TableCell sx={{ fontSize: "15px" }}>{template.subject}</TableCell>
                <TableCell sx={{ fontSize: "15px" }}>
                  <Button onClick={() => handleOpenEdit(template)}>Edit</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Dialog
        open={isEditOpen || isAddOpen}
        onClose={() => {
          setIsEditOpen(false);
          setIsAddOpen(false);
        }}
      >
        <DialogTitle>
          {isEditOpen ? "Edit Template" : "Add New Template"}
        </DialogTitle>
        <DialogContent>
          <TextField
            name="subject"
            label="Subject"
            fullWidth
            margin="dense"
            value={form.subject}
            onChange={handleChange}
          />
          <TextField
            name="body"
            label="Body"
            fullWidth
            multiline
            rows={4}
            margin="dense"
            value={form.body}
            onChange={handleChange}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={isEditOpen ? handleSaveEdit : handleAddTemplate}>
            {isEditOpen ? "Save Changes" : "Add Template"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default EmailTemplates;
