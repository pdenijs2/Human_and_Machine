import pandas as pd
import numpy as np
from wordfreq import word_frequency
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#1. load data

human_raw = pd.read_csv("Data/human_raw.csv")
llm_data = pd.read_csv("Data/Experiment_1_word_and_region_surprisal.csv")

#filter for correct respones only
human_raw = human_raw[human_raw['acierto'] == 1].copy()

print(f"Total after filteing: {len(human_raw)}")

 #2. Map abbrivations to full names

condition_map = {
    'sr': 'Subject',
    'tor': 'Object-Topicalized',
    'orsi': 'Object-Subject-Inversion'
}

#3. word-by-word data

def prepare_word_by_word_data(human_df, llm_df):

    all_data = []
#humans
    for idx, human_row in human_df.iterrows():
        item = human_row['item']
        subj_id = human_row['id']
        cond = human_row['cond']
        nwds = int(human_row['nwds'])

       #skip filllers
        if cond == 'filler':
            continue

        cond_full = condition_map[cond]

         # find same llm data
        llm_row = llm_df[(llm_df['Item'] == item) & (llm_df['Type'] == cond_full)]

        if len(llm_row) == 0:
            continue

        llm_row = llm_row.iloc[0]

        #get text & split into words
        text_words = str(llm_row['Text']).split()

        
        
        #go word by word
        for i in range(nwds):
            word_num = i + 1
            rt_col = f'w{word_num}'
            surprisal_col = f'W{word_num}'

            #find readng time
            rt_val = human_row[rt_col] if rt_col in human_row.index else np.nan

          #get surprisal from LLM
            surprisal_val = llm_row[surprisal_col] if surprisal_col in llm_row.index else np.nan

            # find actual word from 'Text'
            word_text = text_words[i] if i < len(text_words) else ''

            #skip for no reading time
            if pd.isna(rt_val) or rt_val == 0:
                continue

             #skip for missng surprisal
            if pd.isna(surprisal_val):
                continue

            #add everything to all_data
            all_data.append({
                'subject': subj_id,
                'item': item,
                'condition': cond,
                'position': word_num,
                'word': word_text,
                'rt': rt_val,
                'surprisal': surprisal_val
            })

    return pd.DataFrame(all_data)

word_df = prepare_word_by_word_data(human_raw, llm_data)

#check
print(f"Total word obsrvtions: {len(word_df)}")
if len(word_df) > 0:
    print(word_df.head(10))
else:
    print("Error: no df")

#4. Get the predictors

def calculate_log_frequency(word, language='es'):
    word = word.lower().strip()
    #no puncuation
    word = word.strip('.,;:!>')
    freq = word_frequency(word, language)
    if freq == 0:
        freq = 1e-10
    return np.log(freq)

#add word lenght
word_df['length'] = word_df['word'].str.len()

#add frequency
word_df['log_freq'] = word_df['word'].apply(lambda w: calculate_log_frequency(w, 'es'))

#check for freqs
print(word_df[['word', 'length', 'log_freq']].head(10))

#5. Get the spillovr predictors (one and two before)
def add_spillover(df):
    df = df.sort_values(['subject', 'item', 'position']).copy()

#get features for previous two words
    for lag in [1, 2]:
        df[f'length_lag{lag}'] = df.groupby(['subject', 'item'])['length'].shift(lag)
        df[f'log_freq_lag{lag}'] = df.groupby(['subject', 'item'])['log_freq'].shift(lag)
        df[f'surprisal_lag{lag}'] = df.groupby(['subject', 'item'])['surprisal'].shift(lag)

#first words don't have previous words
    lag_cols = [col for col in df.columns if 'lag' in col]
    df[lag_cols] = df[lag_cols].fillna(0)

    return df

#combine into word df
word_df = add_spillover(word_df)

#6. Use mean readign times

agg_df = word_df.groupby(['item', 'condition', 'position', 'word']).agg({
    'rt': 'mean',
    'surprisal': 'first',
    'length': 'first',
    'log_freq': 'first',
    'length_lag1': 'first',
    'log_freq_lag1': 'first',
    'surprisal_lag1': 'first',
    'length_lag2': 'first',
    'log_freq_lag2': 'first',
    'surprisal_lag2': 'first'
}).reset_index()



#check agg_df
print(f"{len(agg_df)}")
print(agg_df.head(10))

#7. Get numpy arrays
#baseline
baseline_predictors = ['length', 'log_freq', 'length_lag1', 'log_freq_lag1', 'length_lag2', 'log_freq_lag2']

#target=with surprisal
target_predictors = baseline_predictors + ['surprisal', 'surprisal_lag1', 'surprisal_lag2']

X_baseline = agg_df[baseline_predictors].values
X_target = agg_df[target_predictors].values
y = agg_df['rt'].values

#check shape
print(f"baseline: {X_baseline.shape}")
print(f"target: {X_target.shape}")

#8. cross-fold & validation function


def cross_val(X, y, n_folds=10, random_state=42):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model = LinearRegression()
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores)

#9 Fit the actual models


r2_baseline = cross_val(X_baseline, y)
r2_target = cross_val(X_target, y)
observed_delta = r2_target - r2_baseline

#check R^2
print(f"Baseline R^2: {r2_baseline}")
print(f"Target -surprisal- R^2: {r2_target}")
print(f"Delta R^2: {observed_delta}")

#10 Permutations Test
#create perumation function

def permutation_test_delta_r2(X_baseline, X_target, y):
    
    #Get r^2 values
    r2_baseline = cross_val(X_baseline, y)
    r2_target = cross_val(X_target, y)
    observed_delta = r2_target - r2_baseline
    
    # do the permutation
    n_permutations = 1000
    perm_deltas = np.zeros(n_permutations)
    
    surprisal_cols = [-3, -2, -1]

     #for each premutation
    for i in range(n_permutations):
        X_permuted = X_target.copy()
        for col_idx in surprisal_cols:
            X_permuted[:, col_idx] = np.random.permutation(X_permuted[:, col_idx])
            
        r2_baseline_perm = cross_val(X_baseline, y)
        r2_target_perm = cross_val(X_permuted, y)
        perm_deltas[i] = r2_target_perm - r2_baseline_perm
        
        #calculate p values
        p_value = np.mean(perm_deltas >= observed_delta)
        
        return observed_delta, p_value

#run the test    
observed_delta, p_value = permutation_test_delta_r2(X_baseline, X_target, y)

print(f"Observed r^2: {observed_delta}")
print(f"p-value: {p_value}")
