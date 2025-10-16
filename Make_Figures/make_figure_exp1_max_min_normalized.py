import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


human_region_csv = "../Data/human_region_rt.csv"
human_raw_csv = "../Data/human_raw.csv"  
llm_csv = "../Data/Experiment_1_region_average_by_type.csv"

# Regions 
regions = ["Region1", "Region2", "Region3", "Region4"]

# Display labels 
region_labels = {
    "Region1": "RCpro",
    "Region2": "RCini",
    "Region3": "RCend",
    "Region4": "MatV",
}

# Map
type_rename = {
    "object-subject-inversin": "OVS",
    "object-subject-inversion": "OVS",
    "object subject inversion": "OVS",
    "object-subject": "OVS",
    "object-topicalized": "OSV",
    "object topicalized": "OSV",
    "subject": "SVO",
    "sr": "SVO",
    "tor": "OSV",
    "orsi": "OVS",
}
type_order = ["OVS", "OSV", "SVO"]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def _as_key_cols(df: pd.DataFrame, cols=("id","item","cond")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError("Required key not found.")
        out[c] = out[c].astype(str).str.strip()
    return out

def _se(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return np.nan if x.size <= 1 else x.std(ddof=1) / np.sqrt(x.size)


def load_correct_keys(raw_csv: str) -> pd.DataFrame:

    raw = pd.read_csv(raw_csv)
    raw = _lower(raw)

    ok = raw[raw["acierto"] == 1].copy()
    ok = _as_key_cols(ok, ("id","item","cond"))
    keys = ok[["id","item","cond"]].drop_duplicates().reset_index(drop=True)
    return keys

#load and summarize humans
def load_and_summarize_humans(region_csv: str, keys_df: pd.DataFrame):

    df = pd.read_csv(region_csv)
    df = _lower(df)

    region_cols = [c for c in df.columns if c.startswith("region") and c.endswith("_rrt")]
    if len(region_cols) != 4:
        raise ValueError("Expected 4 regions")

    df = _as_key_cols(df, ("id","item","cond"))
    before_rows = len(df)
    df = df.merge(keys_df, on=["id","item","cond"], how="inner")
    after_rows = len(df)


    # melt to long
    id_vars = ["id","item","type","cond"]
    long = df.melt(id_vars=id_vars, value_vars=region_cols,
                   var_name="region", value_name="rt")
    long["region"] = long["region"].str.replace("_rrt","",regex=False).str.title()
    long["rt"] = pd.to_numeric(long["rt"], errors="coerce")
    long = long[long["rt"].notna()].copy()

    # map types
    long["type"] = (long["type"].astype(str).str.strip().str.lower()
                    .map(type_rename).fillna(long["type"].astype(str)))
    long["type"] = pd.Categorical(long["type"], categories=type_order, ordered=True)

    # trim
    stats = (long.groupby(["type","region"], observed=False)["rt"]
                  .agg(["mean","std"])
                  .reset_index()
                  .rename(columns={"mean":"grp_mean","std":"grp_sd"}))
    long = long.merge(stats, on=["type","region"], how="left")
    long["cut_upper"] = long["grp_mean"] + 3.5 * long["grp_sd"]
    before_n = len(long)
    long_trim = long[long["rt"] <= long["cut_upper"]].copy()
    after_n = len(long_trim)
    print("Trim")

    # participant-by-condition means
    pmeans = (long_trim.groupby(["id","type","region"], as_index=False)
                      .agg(rt=("rt","mean")))

    # group mean +/- SE across participants
    human_summary = (pmeans.groupby(["type","region"], as_index=False)
                           .agg(mean_ms=("rt","mean"),
                                se_ms=("rt", _se)))
    human_summary["type"] = pd.Categorical(human_summary["type"], categories=type_order, ordered=True)
    human_summary = human_summary.sort_values(["region","type"]).reset_index(drop=True)

    human_long_trim = long_trim[["id","item","type","region","rt"]].copy()
    return human_long_trim, human_summary

# llm surprisal bits
def load_llm_bits_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _lower(df)
    region_cols = [c for c in df.columns if c.startswith("region") and c.endswith("_surprisal")]

    long = df[["item","type"] + region_cols].melt(
        id_vars=["item","type"], value_vars=region_cols,
        var_name="region", value_name="surprisal_nats"
    )
    long["region"] = long["region"].str.replace("_surprisal","",regex=False).str.title()
    long["surprisal_nats"] = pd.to_numeric(long["surprisal_nats"], errors="coerce")
    long = long.dropna(subset=["surprisal_nats"]).copy()

    long["type"] = (long["type"].astype(str).str.strip().str.lower()
                    .map(type_rename).fillna(long["type"].astype(str)))
    long["type"] = pd.Categorical(long["type"], categories=type_order, ordered=True)

    long["surprisal_bits"] = long["surprisal_nats"] / np.log(2.0)

    llm_summary = (long.groupby(["type","region"], as_index=False)
                        .agg(mean_bits=("surprisal_bits","mean"),
                             se_bits=("surprisal_bits", _se)))
    llm_summary["type"] = pd.Categorical(llm_summary["type"], categories=type_order, ordered=True)
    llm_summary = llm_summary.sort_values(["region","type"]).reset_index(drop=True)
    return llm_summary

#min-max normalization
def scale_within_experiment(human_sum: pd.DataFrame, llm_sum: pd.DataFrame):
    hs = human_sum.copy()
    ls = llm_sum.copy()

    #human
    h_min = float(hs["mean_ms"].min())
    h_max = float(hs["mean_ms"].max())
    h_rng = h_max - h_min if (h_max - h_min) != 0 else 1.0
    hs["human_scaled"] = (hs["mean_ms"] - h_min) / h_rng
    hs["human_se_scaled"] = hs["se_ms"] / h_rng

    #llm
    l_min = float(ls["mean_bits"].min())
    l_max = float(ls["mean_bits"].max())
    l_rng = l_max - l_min if (l_max - l_min) != 0 else 1.0
    ls["llm_scaled"] = (ls["mean_bits"] - l_min) / l_rng
    ls["llm_se_scaled"] = ls["se_bits"] / l_rng


    return hs, ls

#Plot
def plot_summary_scaled(human_sum_scaled: pd.DataFrame,
                        llm_sum_scaled: pd.DataFrame,

                        out_png="Figure_Experiment1.png"):
    types = [t for t in type_order
             if (t in set(human_sum_scaled["type"])) or (t in set(llm_sum_scaled["type"]))]
    type_to_x = {t: i+1 for i,t in enumerate(types)}
    jitter_h, jitter_m = -0.12, +0.12

    human_color = "C0"  
    llm_color  = "black"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300, constrained_layout=True)
    axes = axes.flatten()

    for idx, (ax, region) in enumerate(zip(axes, regions)):
        hR = human_sum_scaled[human_sum_scaled["region"] == region]
        lR = llm_sum_scaled[llm_sum_scaled["region"] == region]

        # Human
        xs_h, ys_h, es_h = [], [], []
        for t in types:
            row = hR[hR["type"] == t]
            if len(row):
                r = row.iloc[0]
                xs_h.append(type_to_x[t] + jitter_h)
                ys_h.append(float(r["human_scaled"]))
                es_h.append(float(r["human_se_scaled"]))
        if xs_h:
            ax.errorbar(xs_h, ys_h, yerr=es_h, fmt="o", markersize=4, capsize=4,
                        linestyle="", color=human_color, ecolor=human_color,
                        label="Human (±SE)")

        # LLM
        xs_m, ys_m, es_m = [], [], []
        for t in types:
            row = lR[lR["type"] == t]
            if len(row):
                r = row.iloc[0]
                xs_m.append(type_to_x[t] + jitter_m)
                ys_m.append(float(r["llm_scaled"]))
                es_m.append(float(r["llm_se_scaled"]))
        if xs_m:
            ax.errorbar(xs_m, ys_m, yerr=es_m, fmt="s", markersize=4, capsize=4,
                        linestyle="", color=llm_color, ecolor=llm_color,
                        label="LLM (± item SE)")


        panel_title = region_labels.get(region, region)
        ax.set_title(f"{panel_title}", fontsize=10)
        ax.set_xticks([type_to_x[t] for t in types], labels=types)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.5, len(types) + 0.5)
        ax.grid(alpha=0.25)


        row = idx // 2      
        col = idx % 2
       
        if col == 0:
            ax.set_ylabel("Scaled Measure (Min-Max Normalization)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        
        if row == 1:
            ax.set_xlabel("Sentence type")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

       
        if idx == 0:
            ax.legend(loc="best", frameon=True, fontsize=8)


    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure.png")

def main():

    keys = load_correct_keys(human_raw_csv)
    human_long_trim, human_sum = load_and_summarize_humans(human_region_csv, keys)

    llm_sum = load_llm_bits_summary(llm_csv)
    human_sum_scaled, llm_sum_scaled = scale_within_experiment(human_sum, llm_sum)

    plot_summary_scaled(human_sum_scaled, llm_sum_scaled)

if __name__ == "__main__":
    main()

