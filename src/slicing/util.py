from __future__ import print_function

import json
import os
import os.path
import typing as T
from pprint import pprint

import matplotlib as mpl
import numpy as np
import pandas as pd
import pygsheets as pyg
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# TODO sort when save_all_fits

# === Globals ===
RECORDS = pd.read_csv("data/data_na_disc.csv") # TODO: consider na_keep
VERBOSE = 0

FloatT = (T.Any, float)
ObjectT = (T.Any, object)


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
  return pyg.authorize(custom_credentials=creds)

GOOGLE_CREDS = get_gcreds()

def write_to_sheet(df, sheet, page):
  worksheet = GOOGLE_CREDS.open(sheet).worksheet("index", page)
  worksheet.update_values("A1:Z500", [df.columns.values.tolist()] + df.values.tolist())

def get_format_json(rgb):
  return json.dumps({"backgroundColorStyle": {"rgbColor": {"red": 0, "green": 0, "blue": 1}}})

def format_column(sheet, page, col):
  worksheet = GOOGLE_CREDS.open(sheet).worksheet("index", page)
  start, end = pyg.Address((1, col)).label, pyg.Address((500, col)).label
  vals = worksheet.get_col(col)[1:]
  vals = [val for val in vals if val != ""]
  mn, mx = float(min(vals)), float(max(vals))
  rng = np.linspace(mn, mx, num=5, endpoint=True)
  for i in range(1, 6):
    frmt = get_format_json(0)
    print(frmt)
    worksheet.add_conditional_formatting(start, end, condition_type="NUMBER_BETWEEN", 
                                         format=frmt, condition_values=[rng[i - 1], rng[i]])

# == Misc Helpers ==
def verbose_helper(i, N, num=10):
  """ Helps with verbose progress logs. """
  mults = np.floor(np.linspace(0, N, num + 1, endpoint=True)).astype(int)
  if i in mults:
    return list(mults).index(i)
  else:
    return -1

def get_colors(n):
  if n <= 10:
    return mpl.color_sequences['tab10']
  elif n <= 20:
    return mpl.color_sequences['tab20']
  print("More than 20 colors.")