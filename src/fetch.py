import pandas as pd
import gspread

df_cols = [
  "train set 1",
  "train set 1 size",
  "train set 1 jsd",
  "train set 2",
  "train set 2 size",
  "train set 2 jsd",
  "test set",
  "language from",
  "language to",
  "geographic",
  "genetic",
  "syntactic",
  "phonological",
  "inventory",
  "featural",
  "sp-BLEU"
]
df_dtypes = {
    "train set 1 size": "Int64",
    "train set 1 jsd": "Float64",
    "train set 2 size": "Int64",
    "train set 2 jsd": "Float64",
    "geographic": "Float64",
    "genetic": "Float64",
    "syntactic": "Float64",
    "phonological": "Float64",
    "inventory": "Float64",
    "featural": "Float64"
}

def read_pure_data(gc) -> pd.DataFrame:
  """Reads experiment records from GSheet to a DataFrame."""
  worksheet = gc.open('Experiment 1 Data').get_worksheet(0)
  rows = worksheet.get_all_values()
  df = pd.DataFrame.from_records(rows[4:109], coerce_float=True)
  df.columns = [
      "model",
      "train set 1",
      "train set 1 size",
      "train set 2",
      "train set 2 size",
      "ka-flores",
      "ka-bible",
      "ka-pmo",
      "gu-flores",
      "gu-bible",
      "gu-pmo",
      "hi-flores",
      "hi-bible",
      "hi-pmo",
      "si-flores",
      "si-bible",
      "si-gov",
      "ta-flores",
      "ta-bible",
      "ta-gov"
  ]

  df[df==''] = pd.NA
  df['model'] = 'mbart FT'
  # forward fill the training sets
  df['train set 1'] = df["train set 1"].fillna(method="ffill")
  df['train set 1 size'] = df["train set 1 size"].fillna(method="ffill")
  df['train set 2'] = df["train set 2"].fillna(method="ffill")
  df['train set 2 size'] = df["train set 2 size"].fillna(method="ffill")
  # wherever there is no train set 1, set it to NA
  df.loc[df['train set 1 size'] == "0k", 'train set 1'] = pd.NA
  return df

def read_pure_jsd(gc) -> pd.DataFrame:
  """Reads jsd values from GSheet to a DataFrame."""
  worksheet = gc.open('Experiment 1 Data').get_worksheet(1)
  rows = worksheet.get_all_values()
  df = pd.DataFrame.from_records(rows[1:38], coerce_float=True)
  df.columns = [
      "train set",
      "train set size",
      "test set",
      "test set size",
      "ka",
      "gu",
      "hi",
      "si",
      "ta",
  ]
  df[df==''] = pd.NA
  return df

def read_pure_l2v(gc) -> pd.DataFrame:
  """Reads lang2vec distance values from GSheet to a DataFrame."""
  worksheet = gc.open('Experiment 1 Data').get_worksheet(2)
  rows = worksheet.get_all_values()
  df = pd.DataFrame.from_records(rows[1:7], coerce_float=True)
  df.columns = [
    "distance",
    "kan-eng",
    "guj-eng",
    "hin-eng",
    "sin-eng",
    "tam-eng"
  ]
  df[df==''] = pd.NA
  return df

def find_jsd(jsd_df: pd.DataFrame, train: str, train_size: int, test: str,
             lang: str) -> float:
  """ Returns jsd of a record in jsd_df.
  Arguments:
    jsd_df: DataFrame containing jsds returned from read_pure_jsd.
    train, train_size, test, lang: Values describing a row of jsd_df.
  """
  slice = jsd_df[(jsd_df['train set'] == train) &
                 (jsd_df['train set size'] == train_size) &
                 (jsd_df['test set'] == test)]
  if len(slice) == 0:
    return pd.NA
  return slice.loc[slice.index[0], lang]

def find_l2v(l2v_df: pd.DataFrame, lang: str) -> float:
  lang_map = {
    'ka': 'kan',
    'gu': 'guj',
    'hi': 'hin',
    'si': 'sin',
    'ta': 'tam'
  }
  return l2v_df[lang_map[lang] + "-eng"].to_numpy()

def format_df(data_df: pd.DataFrame, jsd_df: pd.DataFrame, 
              l2v_df: pd.DataFrame, discard_na: bool=False) -> pd.DataFrame:
  """ Formats data into a DataFrame.
  Arguments:
    data_df: DataFrame containing records returned from read_pure_data.
    jsd_df: DataFrame containing jsds returned from read_pure_jsd.
    discard_na: If True, rows which have pd.NA as train1, train2, or test are
             discarded.
  Return Value:
    A dataframe containing all records.
    Each row corresponds to one record.
    The columns are:
      - train set 1 (str): Name of train set 1. Can be cc_align, gov, bible.
      - train set 1 size (Int64): Size of train set 1 in thousands of tokens.
      - train set 1 jsd (Float64): JSD between train set 1 and test set.
                                   (Also depends on train set 1 size.)
      - train set 2 (str): Name of train set 2. Can be bible or gov.
      - train set 2 size (Int64): Size of train set 2 in thousands of tokens.
      - train set 2 jsd (Float64): JSD between train set 2 and test set.
                                   (Also depends on train set 2 size.)
      - test set (str): Name of test set. Can be flores, bible, or gov.
      - language from (str): Source language of MT. Always "en".
      - language to (str): Target language of MT.
                           Can be ka (Kannada), gu (Gujarati), hi (Hindi),
                           si (Sinhala), ta (Tamil).
      - sp-BLEU (Float64): sp-BLEU of experiment record.
    NOTE: Dataset pmo is also represented as gov for simplicity.
  """
  df = pd.DataFrame(columns=df_cols)
  for row in range(1, data_df.shape[0], 2):
    for col in data_df.columns[5:]:
      if pd.isna(data_df[col][row]): # if there is no sp-BLEU score
        continue
      lang = col[:2]
      test = 'pmo/gov' if col[3:].lower() in ['pmo', 'gov'] else col[3:].lower()

      train1 = data_df['train set 1'][row]
      train1 = pd.NA if pd.isna(train1) else train1.lower()
      size1 = "0k" if pd.isna(train1) else data_df['train set 1 size'][row]
      jsd1 = pd.NA if pd.isna(train1) else find_jsd(jsd_df, train1, size1, test, lang)

      train2 = data_df['train set 2'][row]
      train2 = pd.NA if pd.isna(train2) else train2.lower()
      size2 = "0k" if pd.isna(train2) else data_df['train set 2 size'][row]
      jsd2 = pd.NA if pd.isna(train2) else find_jsd(jsd_df, train2, size2, test, lang)

      new_row = {
        "train set 1": train1 if pd.isna(train1) or train1 != 'pmo/gov' else 'gov',
        "train set 1 size": int(size1[:-1]),
        "train set 1 jsd": jsd1,
        "train set 2": train2 if pd.isna(train2) or train2 != 'pmo/gov' else 'gov',
        "train set 2 size": int(size2[:-1]),
        "train set 2 jsd": jsd2,
        "test set": test if pd.isna(test) or test != 'pmo/gov' else 'gov',
        "language from": "en",
        "language to": lang,
        "sp-BLEU": float(data_df[col][row])
      }
      new_row.update(zip(l2v_df["distance"], find_l2v(l2v_df, lang)))
      if discard_na and (pd.isna(train1) or pd.isna(train2) or pd.isna(test)):
        continue
      df.loc[len(df.index)] = new_row
  return df.astype(df_dtypes)

def read_data(discard_na: bool=False) -> pd.DataFrame:
  """ Reads data into a Dataframe of the format returned by format_df. """
  gc = gspread.oauth()
  data_df = read_pure_data(gc)
  jsd_df = read_pure_jsd(gc)
  l2v_df = read_pure_l2v(gc)
  return format_df(data_df, jsd_df, l2v_df, discard_na)

if __name__ == "__main__":
    df = read_data(discard_na=True)
    df.to_csv("data/data_na_disc.csv", index=False)
    df = read_data(discard_na=False)
    df.to_csv("data/data_na_kept.csv", index=False)