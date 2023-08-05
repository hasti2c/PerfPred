from __future__ import print_function

import os
import os.path
import sys
import time
import typing as T
from configparser import ConfigParser
from pprint import pprint

import gspread
import gspread_dataframe as gsdf
import matplotlib as mpl
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# === Config ===
EXPERIMENT_TYPE = "all"
MAX_NVARS, MAX_NSPLITS = 1, 0
DATA_PATH = "data"
CONFIG_FILE = "config.txt"
WRITE_TO_SHEET = False
SHEET_NAMES = {}
SHEETS = {}

def read_config():
    config = ConfigParser()
    config.read(CONFIG_FILE)
    global WRITE_TO_SHEET, SHEET_NAMES, EXPERIMENT_TYPE, DATA_PATH, MAX_NVARS, MAX_NSPLITS
    WRITE_TO_SHEET = config['API']['gsheet'] in ["True", "true", "1"]
    SHEET_NAMES["costs"], SHEET_NAMES["cost stats"] = config['API']['costs sheet'], config['API']['cost stats sheet']
    SHEET_NAMES["baselines"] = config['API']['baselines sheet']
    EXPERIMENT_TYPE = config['Experiment']['type']
    MAX_NVARS, MAX_NSPLITS = int(config['Experiment']['max nvars']), int(config['Experiment']['max nsplits'])
    DATA_PATH = os.path.join(DATA_PATH, EXPERIMENT_TYPE)

read_config()

# === Globals ===
RECORDS = pd.read_csv(os.path.join(DATA_PATH, "records.csv"))

FloatT = (T.Any, float)
ObjectT = (T.Any, object)

COLOR_MAP = mpl.colormaps['rainbow']

# == Google Sheets ==

def get_gcreds():
  # If modifying these scopes, delete the file token.json.
  SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive']
  creds = None
  if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
      token.write(creds.to_json())
  return gspread.authorize(creds)

def init_sheets():
  gc = get_gcreds()
  for sheet, name in SHEET_NAMES.items():
    SHEETS[sheet] = gc.open(name)

def clear_sheet(sh):
  try:
    for wsh in sh.worksheets()[1:]:
      sh.del_worksheet(wsh)
  except gspread.exceptions.APIError:
    print("Sleeping for 60 seconds...", file=sys.stderr)
    time.sleep(60)
    clear_sheet(sh)

def write_to_sheet(df, sh, name=None, index=True):
  try:
    try:
      if name is None:
        wsh = sh.get_worksheet(0)
      else:
        wsh = sh.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
      wsh = sh.get_worksheet(0).duplicate(insert_sheet_index=len(sh.worksheets()), new_sheet_name=name)
    gsdf.set_with_dataframe(wsh, df, include_index=index, include_column_header=True, resize=True)
    if name is not None:
      wsh.update_title(name)
    print(f"Wrote to {sh.title}:{name}.")
  except gspread.exceptions.APIError:
    print("Sleeping for 60 seconds...", file=sys.stderr)
    time.sleep(60)
    write_to_sheet(df, sh, name, index)

if WRITE_TO_SHEET:
  init_sheets()

# == File Helpers ==
def empty_folder(path: str) -> None:
  """ Deletes (deep) all files from the folder described by the path.
  Does not delete subdirectories.
  """
  for f in os.listdir(path):
    f_path = os.path.join(path, f)
    if os.path.isdir(f_path):
      empty_folder(f_path)
    else:
      os.unlink(f_path)