wget -O pretrained_h36m_detectron_coco.bin https://www.dropbox.com/scl/fi/qflj7utmg6klaq6tuxmpb/pretrained_h36m_detectron_coco.bin?rlkey=nmwv4lc1bq8986gngpej38usq
EXPECTED_HASH="d3219e005b50591f694da5cbaf6849f060d6b2cf895864a779f8a992ac63a232"

CALCULATED_HASH=$(sha256sum pretrained_h36m_detectron_coco.bin | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp pretrained_h36m_detectron_coco.bin backend/algorithms/gait_basic/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin
fi

rm pretrained_h36m_detectron_coco.bin

wget -O gait-turn-time.pth https://www.dropbox.com/scl/fi/di8az1o4hgmv85uyg2hw0/gait-turn-time.pth?rlkey=6yxbjauub469ih2xcd82gkqia

EXPECTED_HASH="3878229e15542573a75948999ffff4d75d7dba3d00dbbe358bb2e15270348f08"

# Calculate the SHA-256 hash of the file
CALCULATED_HASH=$(sha256sum gait-turn-time.pth | awk '{ print $1 }')

# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp gait-turn-time.pth backend/algorithms/gait_basic/gait_study_semi_turn_time/weights/semi_vanilla_v2/gait-turn-time.pth
fi

rm gait-turn-time.pth

wget -O gait-depth-weight.pth https://www.dropbox.com/scl/fi/9yop34l6lz1clxzxv6gf0/gait-depth-weight.pth?rlkey=qfbn9kkwk7z4qfegbww5tu13x
EXPECTED_HASH="8cddbd270b9e046981892a1184428268a75d955ba01ce7d4019b59ef332f0000"

CALCULATED_HASH=$(sha256sum gait-depth-weight.pth | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp gait-depth-weight.pth backend/algorithms/gait_basic/depth_alg/weights/gait-depth-weight.pth
fi

rm gait-depth-weight.pth
