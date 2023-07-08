function DataModelChoice({ dataType, handleDataTypeChange, availableDataTypes, modelName, handleModelNameChange, availableModelName }) {

  return (
    <>
      <div className="form-group">
        <label className="col-sm-1 control-label">Data Type</label>
        <div className="col-sm-10">
          <select className="form-control" value={dataType} onChange={handleDataTypeChange}>
            {availableDataTypes.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="form-group">
        <label className="col-sm-1 control-label">Model Name</label>
        <div className="col-sm-10">
          <select className="form-control" value={modelName} onChange={handleModelNameChange}>
            {availableModelName.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>
    </>
  )
}

export default DataModelChoice