from __future__ import print_function

import os
import os.path
import sys
import time
import typing as T
from configparser import ConfigParser
from pprint import pprint

import gspread as gs
import gspread_dataframe as gsdf
import matplotlib as mpl
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# === Config ===
CONFIG_FILE = "config.txt"

EXPERIMENT_TYPE = "all"       # Possible values: "all", "one stage", "two stage".
MIN_POINTS = 0                # Minimum number of points there needs to be in all slices to keep the splits.
DATA_PATH = "data"            # Path of data directory.
RESULTS_DIR = "results"       # Subdirectory for results.
WRITE_TO_SHEET = False        # Whether or not to save data to google sheets. If True, requires credentials.json.
SHEET_NAMES = {}              # Name of sheets. Possible keys: "costs", "cost stats", "baselines"
SHEETS = {}                   # Sheets corresponding to names.

def read_config() -> None:
    """ Reads config variables from config file. """
    config = ConfigParser()
    config.read(CONFIG_FILE)
    global WRITE_TO_SHEET, SHEET_NAMES, EXPERIMENT_TYPE, DATA_PATH, RESULTS_DIR, MIN_POINTS
    WRITE_TO_SHEET = config['API']['gsheet'] in ["True", "true", "1"]
    SHEET_NAMES["costs"], SHEET_NAMES["cost stats"] = config['API']['costs sheet'], config['API']['cost stats sheet']
    SHEET_NAMES["cost table"], SHEET_NAMES["assessment"] = config['API']['cost table sheet'], config['API']['assessment sheet']
    EXPERIMENT_TYPE, RESULTS_DIR = config['Experiment']['type'], config['Experiment']['results directory']
    MIN_POINTS = int(config['Experiment']['min points'])
    DATA_PATH = os.path.join(DATA_PATH, EXPERIMENT_TYPE)

read_config()

# === Globals ===
RECORDS = pd.read_csv(os.path.join(DATA_PATH, "records.csv"))

FloatT = (T.Any, float)
ObjectT = (T.Any, object)

COLOR_MAP = mpl.colormaps['rainbow']

# == Google Sheets ==

def get_gcreds() -> gs.client.Client:
  """ Reads Google API credentials. 
  Must have credentials.json with 'drive' and 'spreadsheets' scopes.
  Visit console.cloud.google.com/apis/dashboard to get the credentials.
  Note: If this function is not working, try deleting token.json (if it exists in your project directory).
  """
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
  return gs.authorize(creds)

def init_sheets():
  """ Initializes SHEETS using names in SHEET_NAMES. """
  gc = get_gcreds()
  for sheet, name in SHEET_NAMES.items():
    SHEETS[sheet] = gc.open(name)

def clear_sheet(sh: gs.Spreadsheet) -> None:
  """ Deletes all worksheets in spreadsheet sh other than the first worksheep (at index 0). """
  try:
    for wsh in sh.worksheets()[1:]:
      sh.del_worksheet(wsh)
  except gs.exceptions.APIError:
    print("Sleeping for 60 seconds...", file=sys.stderr)
    time.sleep(60)
    clear_sheet(sh)

def clear_all_sheets() -> None:
  for sh in SHEETS.values():
    clear_sheet(sh)

def write_to_sheet(df: pd.DataFrame, sh: gs.Spreadsheet, name: str=None, index: bool=True, decimals: int=4) -> None:
  """ Writes df to a worksheet.
  
  == Arguments ==
  df: Dataframe to write to worksheet.
  sh: Spreadsheet.
  name: Name of the worksheet in sh. If None, the initial worksheet will be written to.
  index: Whether or not to write df index to sheet.
  """
  df = df.round(4)
  try:
    try:
      if name is None:
        wsh = sh.get_worksheet(0)
      else:
        wsh = sh.worksheet(name)
    except gs.exceptions.WorksheetNotFound:
      wsh = sh.get_worksheet(0).duplicate(insert_sheet_index=len(sh.worksheets()), new_sheet_name=name)
    gsdf.set_with_dataframe(wsh, df, include_index=index, include_column_header=True, resize=True)
    if name is not None:
      wsh.update_title(name)
    print(f"Wrote to {sh.title}:{name}.")
  except gs.exceptions.APIError:
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