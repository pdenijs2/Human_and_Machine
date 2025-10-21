import re 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt



human_csv = "../Data/experiment3_human_means.csv"
llm_csv = "../Data/Experiment_3_word_and_region_surprisal.csv"

# regions
regions = ["Region1", "Region2", "Region3"]
region_titles = {
    "Region1": "RC Control",
    "Region2": "Critical 2",
    "Region3": "Critical 1",
}

# Types
type_order = ["proHF", "proLF", "fullHF", "fullLF"]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in df.columns]
    return out


def _se(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return np.nan if x.size <= 1 else x.std(ddof=1) / np.sqrt(x.size)

def _clean_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _relabel_type(series: pd.Series) -> pd.Series:
    mapping = {
        "fullnpovs": "fullHF",  
        "fullnposv": "fullLF",  
        "pronposv":  "proHF",   
        "pronpovs":  "proLF",
    }
    out = []
    for v in series.astype(str):
        key = _clean_key(v)
        out.append(mapping.get(key, v))
    return pd.Series(out, index=series.index)


# Load humans 
def load_humans_exp3(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df_l = _lower(df)

    needed = [
        "type",
        "region1_surprisal", "region2_surprisal", "region3_surprisal",
        "se_region1_surprisal", "se_region2_surprisal", "se_region3_surprisal",
    ]

    # Relabel types
    df_l["type"] = _relabel_type(df_l["type"])

    stacks = []
    for region, mcol, ecol in [
        ("Region1", "region1_surprisal", "se_region1_surprisal"),
        ("Region2", "region2_surprisal", "se_region2_surprisal"),
        ("Region3", "region3_surprisal", "se_region3_surprisal"),
    ]:
        sub = df_l[["type", mcol, ecol]].copy()
        sub.columns = ["type", "mean_ms", "se_ms"]
        sub["region"] = region
        stacks.append(sub)

    out = pd.concat(stacks, ignore_index=True)

    cats = [t for t in type_order if t in set(out["type"])] + \
           [t for t in out["type"].unique() if t not in type_order]
    out["type"] = pd.Categorical(out["type"], categories=cats, ordered=True)
    out = out.sort_values(["region", "type"]).reset_index(drop=True)
    return out


# Load LLM
def load_llm_bits_exp3(path: str, type_categories) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_l = _lower(df)


    df_l["type"] = _relabel_type(df_l["type"])

    stacks = []
    for region, col in [
        ("Region1", "region1_surprisal"),
        ("Region2", "region2_surprisal"),
        ("Region3", "region3_surprisal"),
    ]:
        sub = df_l[[ "type", col]].copy()
        sub.columns = [ "type", "surprisal_nats"]
        sub["region"] = region
        stacks.append(sub)
    long = pd.concat(stacks, ignore_index=True)

    long["surprisal_nats"] = pd.to_numeric(long["surprisal_nats"], errors="coerce")
    long = long.dropna(subset=["surprisal_nats"]).copy()
    long["surprisal_bits"] = long["surprisal_nats"] / np.log(2.0)

    out = (long.groupby(["type", "region"], as_index=False)
                .agg(mean_bits=("surprisal_bits", "mean"),
                     se_bits=("surprisal_bits", _se)))
    out["type"] = pd.Categorical(out["type"], categories=list(type_categories), ordered=True)
    out = out.sort_values(["region", "type"]).reset_index(drop=True)
    return out


#  Min Max Normalization
def scale_within_experiment(hum: pd.DataFrame, llm: pd.DataFrame):
    hs = hum.copy()
    ls = llm.copy()

    h_min = float(hs["mean_ms"].min())
    h_max = float(hs["mean_ms"].max())
    h_rng = (h_max - h_min) or 1.0

    hs["human_scaled"] = (hs["mean_ms"] - h_min) / h_rng
    hs["human_upper_scaled"] = ((hs["mean_ms"] + hs["se_ms"]) - h_min) / h_rng
    hs["human_lower_scaled"] = ((hs["mean_ms"] - hs["se_ms"]) - h_min) / h_rng
    hs["human_se_scaled"] = (hs["human_upper_scaled"] - hs["human_lower_scaled"]) / 2

    l_min = float(ls["mean_bits"].min())
    l_max = float(ls["mean_bits"].max())
    l_rng = (l_max - l_min) or 1.0

    ls["llm_scaled"] = (ls["mean_bits"] - l_min) / l_rng
    ls["llm_upper_scaled"] = ((ls["mean_bits"] + ls["se_bits"]) - l_min) / l_rng
    ls["llm_lower_scaled"] = ((ls["mean_bits"] - ls["se_bits"]) - l_min) / l_rng
    ls["llm_se_scaled"] = (ls["llm_upper_scaled"] - ls["llm_lower_scaled"]) / 2

    print("Min Max Normalizgin")
    return hs, ls


# Plotting
def plot_exp3_scaled(hum_sc: pd.DataFrame, llm_sc: pd.DataFrame,
                     out_png="Figure_Experiment3.png"):
    types = list(hum_sc["type"].cat.categories)
    xpos = {t: i+1 for i, t in enumerate(types)}
    jitter_h, jitter_m = -0.12, +0.12

    fig, axes = plt.subplots(1, len(regions), figsize=(5.6*len(regions), 4.2),
                             dpi=300, constrained_layout=True)
    if len(regions) == 1:
        axes = [axes]

    for j, (ax, region) in enumerate(zip(axes, regions)):
        hR = hum_sc[hum_sc["region"] == region]
        lR = llm_sc[llm_sc["region"] == region]

        xs_h, ys_h = [], []
        for t in types:
            row = hR[hR["type"] == t]
            if not row.empty:
                xs_h.append(xpos[t] + jitter_h)
                ys_h.append(float(row.iloc[0]["human_scaled"]))
        if xs_h:
            ax.plot(xs_h, ys_h, "o", color="C0", markersize=4,
                    label="Human (mean)", zorder=3)

        xs_m, ys_m, es_m = [], [], []
        for t in types:
            row = lR[lR["type"] == t]
            if not row.empty:
                xs_m.append(xpos[t] + jitter_m)
                ys_m.append(float(row.iloc[0]["llm_scaled"]))
                es_m.append(float(row.iloc[0]["llm_se_scaled"]))
        if xs_m:
            ax.errorbar(xs_m, ys_m, yerr=es_m, fmt="s", markersize=4, capsize=4,
                        linestyle="", color="black", ecolor="black",
                        label="LLM (± item SE)")

        ax.set_title(f"{region_titles.get(region, region)}", fontsize=18)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.5, len(types) + 0.5)
        ax.set_xticks([xpos[t] for t in types], labels=types, fontsize=18)
        ax.grid(alpha=0.25)

        if j == 0:
            ax.set_ylabel("Min-Max Normalized")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if j == 0:
            ax.legend(loc="best", frameon=True, fontsize=8)


    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure.png")



def main():
    hum = load_humans_exp3(human_csv)
    type_categories = hum["type"].cat.categories
    llm = load_llm_bits_exp3(llm_csv, type_categories)

    hum_sc, llm_sc = scale_within_experiment(hum, llm)
    plot_exp3_scaled(hum_sc, llm_sc)

if __name__ == "__main__":
    main()

