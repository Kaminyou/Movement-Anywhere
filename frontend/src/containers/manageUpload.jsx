import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import swal from "sweetalert";

import UnauthorizedPage from "../components/unauthorizedPage";
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import DataModelChoice from "../components/dataModelChoice"
import ManageUploadRecords from '../components/manageUploadRecords'

function ManageUploadPage({ token }) {
  const [expanded, setExpanded] = useState(false);
  const [toggleFocalLength, setToggleFocalLength] = useState(false);

  const [svoFile, setSVOFile] = useState(null);
  const [txtFile, setTXTFile] = useState(null);
  const [mp4File, setMP4File] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableDataTypes, setAvailableDataTypes] = useState([]);
  const [availableModelName, setAvailableModelName] = useState([]);
  const [dataType, setDataType] = useState(null);
  const [modelName, setModelName] = useState(null);

  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedSize, setUploadedSize] = useState(0);
  const [totalSize, setTotalSize] = useState(0);

  const svoFileInputRef = useRef(null);
  const txtFileInputRef = useRef(null);
  const mp4FileInputRef = useRef(null);

  const [date, setDate] = useState('');
  const [description, setDescription] = useState('');
  const [trialID, setTrialID] = useState('');
  const [height, setHeight] = useState('');
  const [focalLength, setFocalLength] = useState(null);

  const [userList, setUserList] = useState([]);
  const [isManager, setIsManager] = useState(false);

  const [selectedUser, setSelectedUser] = useState('');

  const handleSelectedUserChange = (event) => {
    const newSelectedUser = event.target.value;
    setSelectedUser(newSelectedUser);
  };

  const fetchModelAndData = async () => {
    try {
      const response = await axios.get("/api/info/list/datatypes", {
        headers: { Authorization: 'Bearer ' + token }
      });
      setAvailableDataTypes(response.data.datatypes);
      setDataType(response.data.datatypes[0])
      let dataTypeTemp = response.data.datatypes[0]
      try {
        const response = await axios.get("/api/info/list/modelnames", {
          params: { datatype: dataTypeTemp }, headers: { Authorization: 'Bearer ' + token }
        });
        setAvailableModelName(response.data.modelnames)
        setModelName(response.data.modelnames[0])

        let modelNameTemp = response.data.modelnames[0];
        try {
          const response = await axios.get("/api/info/default/focallength", {
            params: { modelname: modelNameTemp }, headers: { Authorization: 'Bearer ' + token }
          });
          setFocalLength(response.data.focalLength)  
        } catch (error) {
          console.error(error);
        }
      } catch (error) {
        console.error(error);
      }
    } catch (error) {
      console.error(error);
    }
  }

  const fetchModelNames = async () => {
    try {
      const response = await axios.get("/api/info/list/modelnames", {
        params: { datatype: dataType }, headers: { Authorization: 'Bearer ' + token }
      });
      setAvailableModelName(response.data.modelnames)
      setModelName(response.data.modelnames[0])

      let modelNameTemp = response.data.modelnames[0];
      try {
        const response = await axios.get("/api/info/default/focallength", {
          params: { modelname: modelNameTemp }, headers: { Authorization: 'Bearer ' + token }
        });
        setFocalLength(response.data.focalLength)  
      } catch (error) {
        console.error(error);
      }

    } catch (error) {
      console.error(error);
    }
  };

  const fetchFocalLength = async () => {
    try {
      const response = await axios.get("/api/info/default/focallength", {
        params: { modelname: modelName }, headers: { Authorization: 'Bearer ' + token }
      });
      setFocalLength(response.data.focalLength)  
    } catch (error) {
      console.error(error);
    }
  }
  
  useEffect(() => {
    fetchModelAndData()
  }, []);

  useEffect(() => {
    fetchModelNames();
  }, [dataType]);

  useEffect(() => {
    fetchFocalLength();
  }, [modelName]);

  const getUserList = async () => {

    await axios.get("/api/manage/listuser", {
      headers: {Authorization: 'Bearer ' + token}
    })
    .then((res) => {
      setUserList(res.data.currentUsers);
      setIsManager(true);
    })
    .catch((error) => {
      console.error(error);
      setIsManager(false);
    });
  }

  useEffect(() => {
    getUserList();
  }, []);


  useEffect(() => {
    const today = new Date();
    const formattedDate = today.toISOString().substr(0, 10);
    setDate(formattedDate);
  }, []);

  const handleExpand = () => {
    setExpanded(!expanded);
  };

  const handleToggleFocalLength = () => {
    setToggleFocalLength(!toggleFocalLength);
  };

  const handleSVOFileChange = (e) => {
    setSVOFile(e.target.files[0]);
  };

  const handleTXTFileChange = (e) => {
    setTXTFile(e.target.files[0]);
  };

  const handleMP4FileChange = (e) => {
    setMP4File(e.target.files[0]);
  };

  const handleDataTypeChange = (e) => {
    setDataType(e.target.value);
  };

  const handleModelNameChange = (e) => {
    setModelName(e.target.value);
  };

  const handleDateChange = (event) => {
    setDate(event.target.value);
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
  };

  const handleTrialIDChange = (event) => {
    setTrialID(event.target.value);
  }

  const handleHeightChange = (event) => {
    setHeight(event.target.value);
  }

  const handleFocalLengthChange = (event) => {
    setFocalLength(event.target.value);
  }

  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const resetForm = () => {
    setLoading(false);
    setDescription('');
    setTrialID('');
    setHeight('');
    setSVOFile(null);
    setTXTFile(null);
    setMP4File(null);
    setUploadProgress(0);
    setUploadedSize(0);
    setTotalSize(0);

    if (svoFileInputRef.current) {
      svoFileInputRef.current.value = '';
    }

    if (txtFileInputRef.current) {
      txtFileInputRef.current.value = '';
    }
  };

  const isNumeric = (n) => {
    return !isNaN(parseFloat(n)) && isFinite(n);
  }

  const handleSubmit = (e) => {
    e.preventDefault();

    if (selectedUser === '') {
      swal({
        title: "Error",
        text: "Please select an user",
        icon: "error",
      });
      return
    }

    if (date === '') {
      swal({
        title: "Error",
        text: "Trial Date is required",
        icon: "error",
      });
      return
    }

    if (trialID === '') {
      swal({
        title: "Error",
        text: "Trial ID is required",
        icon: "error",
      });
      return
    }

    if (trialID.indexOf(' ') >= 0) {
      swal({
        title: "Error",
        text: "Trial ID cannot contain a space",
        icon: "error",
      });
      return
    }

    const formData = new FormData();
    if (dataType === 'gait_svo_and_txt') {
      if (svoFile === null || txtFile === null) {
        swal({
          title: "Error",
          text: "Both SVO and TXT files are required for DataType gait_svo_and_txt",
          icon: "error",
        });
        return;
      }
      formData.append('svoFile', svoFile);
      formData.append('txtFile', txtFile);

    } else if (dataType === 'gait_mp4') {
      if (mp4File === null ) {
        swal({
          title: "Error",
          text: "MP4 file is required for DataType gait_mp4",
          icon: "error",
        });
        return;
      }
      formData.append('mp4File', mp4File);
      if (height === '') {
        swal({
          title: "Error",
          text: "Height should be provided",
          icon: "error",
        });
        return;
      }

      if (!isNumeric(height)) {
        swal({
          title: "Error",
          text: "A valid height value should be provided",
          icon: "error",
        });
        return;
      }


      if (focalLength === '') {
        swal({
          title: "Error",
          text: "Focal length should be provided",
          icon: "error",
        });
        return;
      }

      if (!isNumeric(focalLength)) {
        swal({
          title: "Error",
          text: "A valid focal length value should be provided",
          icon: "error",
        });
        return;
      }

      if (parseFloat(focalLength) <= 0) {
        swal({
          title: "Error",
          text: "A valid focal length value should be provided",
          icon: "error",
        });
        return;
      }
      formData.append('height', height);
      formData.append('focalLength', focalLength);
    }
    formData.append('dataType', dataType);
    formData.append('modelName', modelName);
    formData.append('date', date);
    formData.append('description', description);
    formData.append('trialID', trialID);

    formData.append('account', selectedUser); // special

    const headers = {
      Authorization: 'Bearer ' + token
    };

    setLoading(true); // Set loading to true to disable the button

    axios.post('/api/manage/upload/gait', formData, {
      headers,
      onUploadProgress: (progressEvent) => {
        setUploadedSize(progressEvent.loaded);
        setTotalSize(progressEvent.total);
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      }
    })
    .then(response => {
      console.log(response.data); // Handle the response from the backend
      swal({
        title: "Success",
        text: "Submit Success!",
        icon: "success",
      });
  
    })
    .catch(error => {
      console.error(error);
      swal({
        title: "Error",
        text: "Submit failed",
        icon: "error",
      });

    })
    .finally(() => {
      resetForm(); // Reset form in both success and error cases
    });
  };

  if (!token) {
    // Render unauthorized page or redirect to unauthorized route
    return (
      <UnauthorizedPage/>
    )
  }

  if (!isManager) {
    // Render unauthorized page or redirect to unauthorized route
    return (
      <p>You are not manager</p>
    )
  }

  return (
    <div className='padding-block'>
      <div className="container">
      <div className="row">
          <div className="col-xs-12 col-md-12">
            <div className="about-text">
              <form onSubmit={handleSubmit} className="form-horizontal">
                <div className="form-group">
                  <label className="col-sm-2 control-label">User</label>
                  <div className="col-sm-9">
                  <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                    <InputLabel id="demo-simple-select-standard-label">Select user</InputLabel>
                    <Select
                    labelId="demo-simple-select-standard-label"
                    id="demo-simple-select-standard"
                    value={selectedUser}
                    onChange={handleSelectedUserChange}
                    label="Gait-parameter"
                    disabled={loading}
                    >
                    <MenuItem value="">
                        <em>None</em>
                    </MenuItem>
                    {userList.map((user, index) => (
                        <MenuItem value={user.subordinate}>{user.subordinate}</MenuItem>

                    ))}
                    </Select>
                  </FormControl>
                  </div>
                  </div>
                  <div className="form-group">
                  <label className="col-sm-2 control-label">Trial Date</label>
                  <div className="col-sm-9">
                    <input
                      type="text"
                      className="form-control"
                      placeholder="Trial Date"
                      value={date}
                      onChange={handleDateChange}
                      disabled={loading}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label className="col-sm-2 control-label">Unique Trial ID</label>
                  <div className="col-sm-9">
                    <input
                      type="text"
                      className="form-control"
                      placeholder="YYYY-MM-DD-PATIENT_ID-TRIAL_ID (suggested. e.g., 2024-01-01-1-1)"
                      value={trialID}
                      onChange={handleTrialIDChange}
                      disabled={loading}
                    />
                  </div>
                </div>
                <div className="form-group">
                <label className="col-sm-2 control-label">Description</label>
                  <div className="col-sm-9">
                    <input
                      type="text"
                      className="form-control"
                      placeholder="Description"
                      value={description}
                      onChange={handleDescriptionChange}
                      disabled={loading}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <div className="col-sm-offset-2 col-sm-9">
                    <button type="button" className="btn btn-secondary" onClick={handleExpand}>
                      {expanded ? 'Hide model details' : 'Show model details'}
                    </button>
                  </div>
                </div>
                {expanded && (
                  <DataModelChoice
                    dataType={dataType}
                    handleDataTypeChange={handleDataTypeChange}
                    availableDataTypes={availableDataTypes}
                    modelName={modelName}
                    handleModelNameChange={handleModelNameChange}
                    availableModelName={availableModelName}
                    disabled={loading}
                  />
                )}
                {dataType === 'gait_svo_and_txt' && (
                  <>
                    <div className="form-group">
                      <label className="col-sm-2 control-label">SVO File</label>
                      <div className="col-sm-9">
                      <input type="file" accept=".svo" onChange={handleSVOFileChange} ref={svoFileInputRef} disabled={loading}/>
                      </div>
                    </div>
                    <div className="form-group">
                    <label className="col-sm-2 control-label">TXT File</label>
                    <div className="col-sm-9">
                      <input type="file" accept=".txt" onChange={handleTXTFileChange} ref={txtFileInputRef} disabled={loading}/>
                      </div>
                    </div>
                  </>
                )}
                {dataType === 'gait_mp4' && (
                  <>
                    <div className="form-group">
                      <label className="col-sm-2 control-label">Height</label>
                      <div className="col-sm-9">
                        <input
                          type="text"
                          className="form-control"
                          placeholder="Patient's height in cm (e.g., 165.5)"
                          value={height}
                          onChange={handleHeightChange}
                          disabled={loading}
                        />
                      </div>
                    </div>
                    <div className="form-group">
                      <div className="col-sm-offset-2 col-sm-9">
                        <button type="button" className="btn btn-secondary" onClick={handleToggleFocalLength}>
                          {toggleFocalLength ? 'Hide settings' : 'More settings'}
                        </button>
                      </div>
                    </div>
                    {toggleFocalLength && (
                      <>
                        <div className="form-group">
                          <label className="col-sm-2 control-label">Camera Focal Length</label>
                          <div className="col-sm-9">
                            <input
                              type="text"
                              className="form-control"
                              placeholder="Camera focal length in mm"
                              value={focalLength}
                              onChange={handleFocalLengthChange}
                              disabled={loading}
                            />
                          </div>
                        </div>
                      </>
                    )}
                    <div className="form-group">
                      <label className="col-sm-2 control-label">MP4 File</label>
                      <div className="col-sm-9">
                    <input type="file" accept="video/mp4" onChange={handleMP4FileChange} ref={mp4FileInputRef} disabled={loading}/>
                    </div>
                    </div>
                  </>
                )}
                <div className="form-group">
                  <div className="col-sm-offset-2 col-sm-9">
                    <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '300px' }}>
                      {loading 
                        ? `Uploading ${uploadProgress}% (${formatBytes(uploadedSize)} / ${formatBytes(totalSize)})` 
                        : 'Upload'}
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
          <div className="col-xs-12 col-md-12">
          <h4>Upload records</h4>
            <ManageUploadRecords token={token}/>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ManageUploadPage
