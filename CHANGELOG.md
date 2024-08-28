# Changelog
## v0.9.8
### Improvements
- Add unit in result api call
- Add more workers

### Fix bugs
- Fix the wrong focal length api call in the upload page

## v0.9.7
### Improvements
- Change the uncertainty representation of `tools/find_hard_examples.py`


## v0.9.6
### Improvements
- Add `requestUUID` to report


## v0.9.5
### Improvements
- Make the line clear in visualization (2D alg)


## v0.9.4
### Improvements
- Write out result json


## v0.9.3
### New features
- Add the source code link (GitHub link) in frontend pages
- Improve the demo page

## v0.9.2
### Improvements
- Change all `submitUUID` to `requestUUID` to sync the usage


## v0.9.1
### New features
- Subtask registration


## v0.9.0
### New features
- Add active learning tools


## v0.8.7
### Improvements
- Upgrade test env
- Add unit tests


## v0.8.6
### Improvements
- Improve 2d alg visualization (remove uncertain keypoints/ add target box)
- Change 2d alg video generation task name


## v0.8.5
### New features
- Fully implement distributed inference for the 3D algorithm
- Main worker now can accept multiple tasks

### Improvements
- Update pytest default settings

### Fix bugs
- Fix folder uploading in the synchronizer


## v0.8.4
### New features
- Partially implement distributed inference for the 3D algorithm


## v0.8.3
### Improvements
- Make 2D integration test less strict considering precisions of different GPU fp 


## v0.8.2
### Improvements
- Linting for `backend/algorithms/gait_basic/video2d_analyzer.py`


## v0.8.1
### Improvements
- Improve backend `algorithms.gait_basic.utils.make_video.count_frames`
- Make the `depth_estimation` and `video_generation` steps in 2D algorithm run in parallel


## v0.8.0
### New features
- Implement distributed inference for the 2D algorithm


## v0.7.2
### New features
- Add 3d integration tests

### Improvements
- Improve the analyzer structure
- Linting for `gait_basic`


## v0.7.1
### New features
- Add 2d integration tests
- Add a synchronizer

### Improvements
- Improve the usage of `celery`
- Improve func import in the SVO algorithm


## v0.7.0
### New features
- Add a file server

### Improvements
- Enable storing uploaded/result files at any path
- Add `trial ID` and `description` to the downloaded report

## v0.6.1
### Improvements
- A toggle for `focal length` input text box in frontend 


## v0.6.0
### New features
- Support focal length adjustment

### Improvements
- Change `pretrained_path` to `turn_time_pretrained_path`
- Change `simple_inference` to `turn_time_simple_inference`
- Beautify frontend request pages


## v0.5.0
### New features
- Make `docker socket` path adjustable by `.env`
- Add depth pre-trained model auto-downloading in `setup.sh`

### Improvements
- Improve `CUDA_VISIBLE_DEVICES` settings
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
