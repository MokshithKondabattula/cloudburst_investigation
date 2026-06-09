import argparse
import os
import re
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
from PIL import Image


WV_MIN=200.0
WV_MAX=280.0

P_MIN=0.0
P_MAX=50.0


M_R0=300
M_R1=1279

M_C0=350
M_C1=1241


I_LAT0=959
I_LAT1=1279

I_LON0=2459
I_LON1=2779


FINAL_SIZE=(320,320)


def extract_insat_timestamp(name):

 m=re.search(
  r'(\d{2})([A-Z]{3})(\d{4})_(\d{4})',
  name
 )

 if not m:
  return None

 d,mon,y,t=m.groups()

 months={
  'JAN':'01',
  'FEB':'02',
  'MAR':'03',
  'APR':'04',
  'MAY':'05',
  'JUN':'06',
  'JUL':'07',
  'AUG':'08',
  'SEP':'09',
  'OCT':'10',
  'NOV':'11',
  'DEC':'12'
 }

 ts=f"{y}{months[mon]}{d}{t}"

 return datetime.strptime(
  ts,
  "%Y%m%d%H%M"
 )


def extract_imerg_timestamp(name):

 m=re.search(
  r'(\d{8})-S(\d{2})(\d{2})(\d{2})',
  name
 )

 if not m:
  return None

 d,hh,mm,ss=m.groups()

 ts=datetime.strptime(
  f"{d}{hh}{mm}",
  "%Y%m%d%H%M"
 )

 return ts


def datetime_to_string(dt):

 return dt.strftime("%Y%m%d%H%M")


def normalise_wv(arr):

 arr=np.nan_to_num(
  arr,
  nan=WV_MAX
 )

 arr=np.clip(
  arr,
  WV_MIN,
  WV_MAX
 )

 arr=(
  arr-WV_MIN
 )/(
  WV_MAX-WV_MIN
 )

 arr=1.0-arr

 arr=np.power(
  arr,
  0.7
 )

 return (arr*255).astype(np.uint8)


def normalise_precip(arr):

 arr=np.nan_to_num(
  arr,
  nan=P_MIN
 )

 arr=np.clip(
  arr,
  P_MIN,
  P_MAX
 )

 arr=(
  arr-P_MIN
 )/(
  P_MAX-P_MIN
 )

 arr=np.power(
  arr,
  0.7
 )

 return (arr*255).astype(np.uint8)


def resize_image(arr):

 img=Image.fromarray(
  arr,
  mode='L'
 )

 img=img.resize(
  FINAL_SIZE,
  Image.Resampling.BILINEAR
 )

 return np.array(img)


def save_image(arr,path):

 Image.fromarray(
  arr,
  mode='L'
 ).save(path)


def read_insat_wv(fp):

 with h5py.File(fp,'r') as f:

  raw=f["IMG_WV"][0]

  lut=f["IMG_WV_TEMP"][:]

  counts=np.clip(
   raw,
   0,
   1023
  )

  wv=lut[counts].astype(np.float32)

  wv_crop=wv[
   M_R0:M_R1,
   M_C0:M_C1
  ]

#   wv_crop=np.flipud(
#    wv_crop
#   )

  return wv_crop


def read_imerg(fp):

 try:

  with h5py.File(fp,'r') as f:

   g=f["Grid"]

   if "precipitation" in g:
    field="precipitation"

   elif "precipitationCal" in g:
    field="precipitationCal"

   else:
    return None

   raw=g[field][0]

   precip=raw[
    I_LON0:I_LON1,
    I_LAT0:I_LAT1
   ]

   precip=precip.T

   precip=np.flipud(
    precip
   )

   precip[precip<-9000]=0.0
   precip[precip<0]=0.0

   return precip.astype(np.float32)

 except Exception as e:

  print(f"\nBad IMERG file: {fp}")
  print(e)

  return None


def run_pipeline(
 insat_path,
 imerg_path,
 event_dir,
 timestamp
):

 wv_dir=os.path.join(
  event_dir,
  "wv_images"
 )

 precip_dir=os.path.join(
  event_dir,
  "precipitation_images"
 )

 os.makedirs(
  wv_dir,
  exist_ok=True
 )

 os.makedirs(
  precip_dir,
  exist_ok=True
 )

 wv=read_insat_wv(
  insat_path
 )

 precip=read_imerg(
  imerg_path
 )

 if precip is None:
  return

 wv_img=normalise_wv(
  wv
 )

 precip_img=normalise_precip(
  precip
 )

 wv_img=resize_image(
  wv_img
 )

 precip_img=resize_image(
  precip_img
 )

 event_name=Path(
  event_dir
 ).name

 outname=f"{event_name}_{timestamp}.png"

 wv_path=os.path.join(
  wv_dir,
  outname
 )

 precip_path=os.path.join(
  precip_dir,
  outname
 )

 save_image(
  wv_img,
  wv_path
 )

 save_image(
  precip_img,
  precip_path
 )

 print(f"\nSaved:")
 print(wv_path)
 print(precip_path)


def process_event(event_dir):

 mosdac=os.path.join(
  event_dir,
  "mosdac"
 )

 imerg=os.path.join(
  event_dir,
  "imerg"
 )

 if not os.path.exists(mosdac):
  return

 if not os.path.exists(imerg):
  return

 insat_files=[
  x for x in os.listdir(mosdac)
  if x.lower().endswith(
   (
    ".h5",
    ".hdf5"
   )
  )
 ]

 imerg_files=[
  x for x in os.listdir(imerg)
  if x.lower().endswith(
   (
    ".hdf5",
    ".hdf"
   )
  )
 ]

 insat_map={}
 imerg_map={}

 for f in insat_files:

  ts=extract_insat_timestamp(f)

  if ts:
   insat_map[ts]=os.path.join(
    mosdac,
    f
   )

 for f in imerg_files:

  ts=extract_imerg_timestamp(f)

  if ts:
   imerg_map[ts]=os.path.join(
    imerg,
    f
   )

 common=sorted(
  set(insat_map.keys()) &
  set(imerg_map.keys())
 )

 print(f"\n{Path(event_dir).name}")
 print(f"Matched timestamps: {len(common)}")

 for ts in common:

  try:

   run_pipeline(
    insat_map[ts],
    imerg_map[ts],
    event_dir,
    datetime_to_string(ts)
   )

  except Exception as e:

   print(f"\nFAILED: {ts}")
   print(e)


def main():

 parser=argparse.ArgumentParser()

 parser.add_argument(
  "--root",
  required=True
 )

 args=parser.parse_args()

 events=[]

 for n in range(4,21):

  if n==1:
   event_name="1st_event"

  elif n==2:
   event_name="2nd_event"

  elif n==3:
   event_name="3rd_event"

  else:
   event_name=f"{n}th_event"

  event_path=os.path.join(
   args.root,
   event_name
  )

  if os.path.exists(event_path):

   events.append(event_path)

 for e in events:

  process_event(e)


if __name__=="__main__":
 main()