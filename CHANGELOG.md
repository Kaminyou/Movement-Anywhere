# Changelog
## v0.5.0
### New features
- Make `docker socket` path adjustable by `.env`
- Add depth pre-trained model auto-downloading in `setup.sh`

### Improvements
- Improve CUDA_VISIBLE_DEVICES` settings
- Improve `container run` client settings
- Improve the naming of model weights
- Enable `setup.sh` to check if all the required docker images exist or not

### Fix bugs
- Fix `Werkzeug` version bug (conflict with `Flask`)


## v0.4.0
### New features
- Support a 2D only algorithm with a mp4 video and height inputs


## v0.3.1
### New features
- Retry and timeout setting for `svo export` and `depth sensing` in `SVOGaitAnalyzer` due to unknown occasional errors

### Improvements
- Remove old gait algorithm `BasicGaitAnalyzer`


## v0.3.0
### New features
- Add a **researcher** category
    - Researcher belongs to one manager
    - Researcher cannot upload record
    - Researcher can see records of its managers'
    - Researcher can only see black background videos

### Improvement
- Other category (manager/general) can see the video with human and real background

## v0.2.2
### New features
- To protect patient's privacy, the default video change to a black background one with full keypoints shown. The setting can be changed by modify the `get_video` api in `backend/routers/user.py`

## v0.2.1
### Improvement
- Add an automatical timestamp txt fixing logic

## v0.2.0
### Improvement
- Enable raw svo and txt uploading
- Add algorithms to remove any non-targeted person
- Make Rscript execution independent between jobs (not computing in the source code folder)
- Make 3D estimation execution independent between jobs (not computing in the source code folder)

### New features
- Examine each field in frontend during uploading
- Block any modification in each field during uploading
- Enable providing a unique trial ID
- Show uploading progress

## v0.1.1
### New features
- Provide instruction for 3D trajetories conversion
- Provide solely 2D vedio inference mode
- Add a verifier for detected keypoints

## v0.1.0
### Initial version
- Support the algorithms mentioned in the publication
