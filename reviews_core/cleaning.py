import pandas as pd
import numpy as np
import re

def cleaning(df_raw: pd.DataFrame):
    """Clean raw Google Play reviews into a consistent schema."""
    df = df_raw.copy()

    # Drop columns we don't keep
    if 'user_name' in df.columns:
        df = df.drop(columns=['user_name'])

    # Create sequential review_id starting at 1
    df.insert(0, 'review_id', np.arange(1, 1 + len(df), dtype='int32'))

    # Parse date columns
    for c in ['date', 'Reply_Date']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    
    # Create column 'year'
    df['year'] = df['date'].dt.year.astype('Int16')
    
    # Create column 'replied' where 1 means replied and 0 means not replied
    df['replied'] = reviews['Reply'].notna().astype('int8')
    
    # Create 'time_to_reply(h)'
    delta = df['Reply_Date'] - df['date']
    df['time_to_reply(h)'] = (delta.dt.total_seconds() / 3600).round(2)
    
    # Split 'app_version''app_version_head','app_version_detail'
    vs = df['App_Version'].astype(str).str.split('.', n=1, expand=True)
    df['app_version_head'] = vs[0]
    df['app_version_detail'] = vs[1]
    
    # rows with no '.' → put the whole value in version_detail, macro = NA
    no_dot = ~df['App_Version'].astype(str).str.contains('.', regex=False)
    df.loc[no_dot, 'app_version_head'] = pd.NA
    df.loc[no_dot, 'app_version_detail'] = df.loc[no_dot, 'App_Version']
    # drop legacy column after splitting
    df = df.drop(columns=['App_Version'])
    
    # Rename columns
    colmap = {
        'app_name': 'app',
        'text': 'review_text',
        'date': 'review_date',
        'Reply': 'reply_text',
        'Reply_Date': 'reply_date',
    }
    df = df.rename(columns=colmap)

    # --- text cleaning (creates *_clean) ---
    def _clean_text(x):
        if not isinstance(x, str):
            return pd.NA
        t = x.lower()                                # lowercase
        t = re.sub(r"http\S+", "", t)                # remove URLs
        t = re.sub(r"\s+", " ", t).strip()           # normalize whitespace
        return t if t else pd.NA

    df['review_text_clean'] = df['review_text'].map(_clean_text)
    
    df['reply_text_clean']  = df['reply_text'].map(_clean_text)
    
    # Reorder columns; keep any others at the end
    new_order = [
        'review_id', 'app', 'score','review_text','review_text_clean','review_date','year','thumbs_up','replied','reply_text',
        'reply_text_clean','reply_date','time_to_reply(h)','app_version_head','app_version_detail'
    ]

    df = df[new_order]

    return df