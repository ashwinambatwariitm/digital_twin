#!/bin/bash
echo "Downloading INSAT data from MOSDAC..."

lftp -u ashvinambatwar,Hannover@12345 sftp://download.mosdac.gov.in << 'LFTP'
set sftp:auto-confirm yes
set net:max-retries 5
set net:reconnect-interval-base 5
set net:reconnect-interval-multiplier 1
mirror --parallel=4 --verbose . /home/studinstru/ISRO/digital_twin/data/raw/insat/raw/
bye
LFTP

echo "Download complete!"
