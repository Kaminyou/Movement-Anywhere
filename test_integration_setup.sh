wget -O 2024-05-04-1-14.mp4 https://www.dropbox.com/scl/fi/lhqayqkiq9uv7gg9b7b8r/2024-05-04-1-14.mp4?rlkey=nma5rcfmubrxqt4h935jzyht6
EXPECTED_HASH="9aebd7f85e91f5c3818382af67592f7514eedf077fee30c8509dd2474a728ae5"

CALCULATED_HASH=$(sha256sum 2024-05-04-1-14.mp4 | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp 2024-05-04-1-14.mp4 test_data/2024-05-04-1-14.mp4
fi

rm 2024-05-04-1-14.mp4

wget -O 2024-05-04-1-14.txt https://www.dropbox.com/scl/fi/zi3n72rbdfvah4cib2rit/2024-05-04-1-14.txt?rlkey=webyp1tndacryjx0568euhb40
EXPECTED_HASH="5fcab3b698d90a0afdd3270fed13662ad0786ac845c9da80f137648082bb9704"

CALCULATED_HASH=$(sha256sum 2024-05-04-1-14.txt | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp 2024-05-04-1-14.txt test_data/2024-05-04-1-14.txt
fi

rm 2024-05-04-1-14.txt

wget -O 2024-05-04-1-14.svo https://www.dropbox.com/scl/fi/l59fvxp3uznqn8dg7lpsj/2024-05-04-1-14.svo?rlkey=a29mai43flggih17g52f3f92r
EXPECTED_HASH="b0151c0c22dd441a37beb7665a70bc6d07f91d84972c206529136e44aa35226f"

CALCULATED_HASH=$(sha256sum 2024-05-04-1-14.svo | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  cp 2024-05-04-1-14.svo test_data/2024-05-04-1-14.svo
fi

rm 2024-05-04-1-14.svo