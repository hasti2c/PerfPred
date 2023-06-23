from __future__ import print_function

import os
import os.path
import typing as T
from pprint import pprint

import gspread
import matplotlib as mpl
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# TODO reorganize files
# TODO fix imports
# TODO repr and str for different classes
# TODO sort when save_all_fits

# === Globals ===
RECORDS = pd.read_csv("data/data_na_disc.csv") # TODO: consider na_keep

VERBOSE = 1

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
  return gspread.authorize(creds)

def write_to_sheet(df, sheet, page):
  gc = get_gcreds()
  worksheet = gc.open(sheet).get_worksheet(page)
  worksheet.update([df.columns.values.tolist()] + df.values.tolist())

# == Misc Helpers ==
def verbose_helper(i, N, num=10):
  """ Helps with verbose progress logs. """
  mults = np.floor(np.linspace(0, N, num + 1, endpoint=True)).astype(int)
  if i in mults:
    return list(mults).index(i)
  else:
    return -1

def get_colors(n): # TODO
  if n > 10:
    print("more than 10 colours")
    return
  cm = mpl.color_sequences['tab10']
  return cm

GREEK = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 
         'iota', 'kappa', 'lamda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 
         'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']