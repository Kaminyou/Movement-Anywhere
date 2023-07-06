import React, { useState, useEffect } from "react";
import axios from "axios";
import UnauthorizedPage from "../components/unauthorizedPage";
import SimpleDashboard from '../components/simpleDashboard'


function DashBoardPage({ token }) {

  const [name, setName] = useState('');
  const [gender, setGender] = useState('');
  const [birthday, setBirthday] = useState('');

  const [diagnose, setDiagnose] = useState('');
  const [stage, setStage] = useState('');
  const [dominantSide, setDominantSide] = useState('');
  const [lded, setLDED] = useState('');
  const [description, setDescription] = useState('');
  const [results, setResults] = useState([]);

  const fetchDefault = async () => {
    try {
      const response = await axios.get("/api/user/profile/personal", {
        headers: { Authorization: 'Bearer ' + token }
      });
      const { name, gender, birthday, diagnose, stage, dominantSide, lded, description } = response.data.profile;
      setName(name);
      setGender(gender);
      setBirthday(birthday);
      setDiagnose(diagnose);
      setStage(stage);
      setDominantSide(dominantSide);
      setLDED(lded);
      setDescription(description);
    } catch (error) {
      console.error(error);
    }
  };
  
  useEffect(() => {
    fetchDefault();
  }, []);

  const fetchResults = async () => {
    await axios.get("/api/user/request/results", {
      headers: {Authorization: 'Bearer ' + token}
    })
    .then((res) => {
      setResults(res.data.results)
    })
    .catch((error) => {
      console.error(error);
    });
  }

  useEffect(() => {
    fetchResults();
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      fetchDefault();
      fetchResults();
      }, 2000);
    return () => clearInterval(interval);
  }, [])


  if (!token) {
    // Render unauthorized page or redirect to unauthorized route
    return (
      <UnauthorizedPage/>
    )
  }

  return (
    <div className="padding-block">
    <div className="container">
      <div className="row">
        <div className="col-md-3">
          <h3>Patient Profile</h3>
          <div className="panel panel-default">
            <div className="panel-body">
              <h4>Basic information</h4>
              <p><strong>Name:</strong> {name}</p>
              <p><strong>Gender:</strong> {gender}</p>
              <p><strong>Birthday:</strong> {birthday}</p>
            </div>
          </div>
          <div className="panel panel-default">
            <div className="panel-body">
              <h4>Diagnosis</h4>
              <p><strong>Diagnose:</strong> {diagnose}</p>
              <p><strong>Stage:</strong> {stage}</p>
            </div>
          </div>
          <div className="panel panel-default">
            <div className="panel-body">
              <h4>Additional Information</h4>
              <p><strong>Dominant Side:</strong> {dominantSide}</p>
              <p><strong>LDED:</strong> {lded}</p>
              <p><strong>Description:</strong> {description}</p>
            </div>
          </div>
        </div>
        <SimpleDashboard
          name={name}
          gender={gender}
          birthday={birthday}
          diagnose={diagnose}
          stage={stage}
          dominantSide={dominantSide}
          lded={lded}
          description={description}
          results={results}
          token={token}
        />
      </div>
    </div>
  </div>
  )
}

export default DashBoardPage