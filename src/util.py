from __future__ import print_function

import os
import os.path
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

# TODO sort when save_all_fits

# === Config ===
EXPERIMENT_TYPE = "all"
DATA_PATH = "data"
CONFIG_FILE = "config.txt"
WRITE_TO_SHEET = False
FITS_SHEET_NAME = "fits"
RESULTS_SHEET_NAME, RESULTS_PAGE = "results", 0
COSTS_SHEET_NAME = "costs"

def read_config():
    config = ConfigParser()
    config.read(CONFIG_FILE)
    global WRITE_TO_SHEET, COSTS_SHEET_NAME, FITS_SHEET_NAME, RESULTS_SHEET_NAME, RESULTS_PAGE, EXPERIMENT_TYPE, \
           DATA_PATH
    WRITE_TO_SHEET = config['API']['gsheet'] in ["True", "true", "1"]
    COSTS_SHEET_NAME, FITS_SHEET_NAME = config['API']['costs sheet'], config['API']['fits sheet']
    RESULTS_SHEET_NAME, RESULTS_PAGE = config['API']['results sheet'], int(config['API']['results page'])
    EXPERIMENT_TYPE = config['Experiment']['type']
    DATA_PATH = os.path.join(DATA_PATH, EXPERIMENT_TYPE)

read_config()

# === Globals ===
RECORDS = pd.read_csv(os.path.join(DATA_PATH, "records.csv"))

FloatT = (T.Any, float)
ObjectT = (T.Any, object)

COLOR_MAP = mpl.colormaps['rainbow']

# == File & GSheet Helpers ==
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

if WRITE_TO_SHEET:
  GSPREAD_CREDS = get_gcreds()
  FITS_SHEET = GSPREAD_CREDS.open(FITS_SHEET_NAME)
  RESULTS_SHEET = GSPREAD_CREDS.open(RESULTS_SHEET_NAME)
  COSTS_SHEET = GSPREAD_CREDS.open(COSTS_SHEET_NAME)

def clear_sheet(sh):
  for wsh in sh.worksheets()[1:]:
    sh.del_worksheet(wsh)

def write_to_sheet(df, sh, page, name=None, index=True):
  try:
    wsh = sh.get_worksheet(page)
  except gspread.exceptions.WorksheetNotFound:
    wsh = sh.get_worksheet(0).duplicate(page)
  gsdf.set_with_dataframe(wsh, df, include_index=index, include_column_header=True, resize=True)
  if name is not None:
    wsh.update_title(name)
  print(f"Wrote to {sh.title}:{name}.")