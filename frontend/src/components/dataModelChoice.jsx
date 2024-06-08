function DataModelChoice({ dataType, handleDataTypeChange, availableDataTypes, modelName, handleModelNameChange, availableModelName, disabled }) {

  return (
    <>
      <div className="form-group">
        <label className="col-sm-2 control-label">Data Type</label>
        <div className="col-sm-9">
          <select className="form-control" value={dataType} onChange={handleDataTypeChange} disabled={disabled}>
            {availableDataTypes.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="form-group">
        <label className="col-sm-2 control-label">Model Name</label>
        <div className="col-sm-9">
          <select className="form-control" value={modelName} onChange={handleModelNameChange} disabled={disabled}>
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