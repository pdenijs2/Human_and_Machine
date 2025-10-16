import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

human_means_csv = "../Data/experiment2_human_means.csv"
llm_items_csv   = "../Data/Experiment_2_word_and_region_surprisal.csv"


# Only plot the last two regions
regions = ["Region3", "Region4"]
region_labels = {"Region1": "Verb", "Region2": "Adverb", "Region3": "Critical", "Region4": "Final"}

# clean up lables?
type_rename = {
 
}
type_order = None

human_color = "C0"   
llm_color   = "black"
marker_size = 5
ecap        = 4
dpi         = 300


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _se(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)


def load_human_means(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _norm_cols(df)

    cmap = {c.lower(): c for c in df.columns}
    if "type" not in cmap:
        raise ValueError("Human CSV must contain a 'Type' column.")

    region_cols = []
    for r in regions:
        key = f"{r}_Surprisal".lower()
        if key not in cmap:
            raise ValueError(f"Human CSV missing column '{r}_Surprisal'.")
        region_cols.append(cmap[key])

    keep = [cmap["type"]] + region_cols
    w = df[keep].copy().rename(columns={cmap["type"]: "type"})

    if type_rename:
        w["type"] = w["type"].map(lambda x: type_rename.get(x, x))

    long = w.melt(id_vars=["type"], value_vars=region_cols,
                  var_name="region_col", value_name="human_mean_ms")
    long["region"] = long["region_col"].str.replace("_Surprisal", "", regex=False)
    long.drop(columns=["region_col"], inplace=True)

    long["human_mean_ms"] = pd.to_numeric(long["human_mean_ms"], errors="coerce")
    long = long.dropna(subset=["human_mean_ms"]).copy()

    types_seen = list(pd.Index(w["type"]).drop_duplicates())
    order = type_order if type_order else types_seen
    long["type"] = pd.Categorical(long["type"], categories=order, ordered=True)

    return long.sort_values(["region", "type"]).reset_index(drop=True)


def load_llm_bits_summary(path: str, type_categories) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _norm_cols(df)

    cmap = {c.lower(): c for c in df.columns}
    for req in ["type", "item"]:
        if req not in cmap:
            raise ValueError(f"LLM CSV must contain a '{req}' column.")
    type_col = cmap["type"]
    item_col = cmap["item"]

    rcols = {}
    for r in regions:
        key = f"{r}_Surprisal".lower()
        if key not in cmap:
            raise ValueError(f"LLM CSV missing column '{r}_Surprisal'.")
        rcols[r] = cmap[key]

    long = df[[item_col, type_col] + list(rcols.values())].melt(
        id_vars=[item_col, type_col],
        value_vars=list(rcols.values()),
        var_name="region_col", value_name="surprisal_nats"
    )
    long.rename(columns={item_col: "item", type_col: "type"}, inplace=True)
    long["region"] = long["region_col"].str.replace("_Surprisal", "", regex=False)
    long.drop(columns=["region_col"], inplace=True)

    if type_rename:
        long["type"] = long["type"].map(lambda x: type_rename.get(x, x))

    long["surprisal_nats"] = pd.to_numeric(long["surprisal_nats"], errors="coerce")
    long = long.dropna(subset=["surprisal_nats"]).copy()
    long["surprisal_bits"] = long["surprisal_nats"] / np.log(2.0)

    long["type"] = pd.Categorical(long["type"], categories=type_categories, ordered=True)

    out = (long.groupby(["type", "region"], as_index=False)
                .agg(mean_bits=("surprisal_bits", "mean"),
                     se_bits=("surprisal_bits", _se),
                     n_items=("surprisal_bits", "size")))
    return out.sort_values(["region", "type"]).reset_index(drop=True)

# min-max noramlizaiton
def scale_within_experiment(hum: pd.DataFrame, llm: pd.DataFrame):
    hs = hum.copy()
    ls = llm.copy()

    h_min = float(hs["human_mean_ms"].min())
    h_max = float(hs["human_mean_ms"].max())
    h_rng = (h_max - h_min) or 1.0
    hs["human_scaled"] = (hs["human_mean_ms"] - h_min) / h_rng

    l_min = float(ls["mean_bits"].min())
    l_max = float(ls["mean_bits"].max())
    l_rng = (l_max - l_min) or 1.0

    ls["llm_scaled"] = (ls["mean_bits"] - l_min) / l_rng
    ls["llm_upper_scaled"] = ((ls["mean_bits"] + ls["se_bits"]) - l_min) / l_rng
    ls["llm_lower_scaled"] = ((ls["mean_bits"] - ls["se_bits"]) - l_min) / l_rng
    ls["llm_se_scaled"]    = (ls["llm_upper_scaled"] - ls["llm_lower_scaled"]) / 2

    return hs, ls

# plot
def plot_exp2_scaled(hum_sc: pd.DataFrame, llm_sc: pd.DataFrame,
                     out_png="Figure_Experiment2.png"):
    types = [t for t in hum_sc["type"].cat.categories if t in set(hum_sc["type"]) or t in set(llm_sc["type"])]
    xmap = {t: i + 1 for i, t in enumerate(types)}
    jitter_h, jitter_m = -0.12, +0.12

    n_panels = len(regions)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 4),
        dpi=dpi,
        constrained_layout=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for j, (ax, rcode) in enumerate(zip(axes, regions)):
        panel_name = region_labels.get(rcode, rcode)
        H = hum_sc[hum_sc["region"] == rcode]
        L = llm_sc[llm_sc["region"] == rcode]

        xs, ys = [], []
        for t in types:
            row = H[H["type"] == t]
            if len(row):
                xs.append(xmap[t] + jitter_h)
                ys.append(float(row.iloc[0]["human_scaled"]))
        if xs:
            ax.plot(xs, ys, "o", markersize=marker_size, color=human_color, label="Human (mean)")

        xs_m, ys_m, es_m = [], [], []
        for t in types:
            row = L[L["type"] == t]
            if len(row):
                xs_m.append(xmap[t] + jitter_m)
                ys_m.append(float(row.iloc[0]["llm_scaled"]))
                es = row.iloc[0]["llm_se_scaled"]
                es_m.append(np.nan if pd.isna(es) else float(es))
        if xs_m:
            ax.errorbar(xs_m, ys_m, yerr=es_m, fmt="s", markersize=marker_size, capsize=ecap,
                        linestyle="", color=llm_color, ecolor=llm_color, label="LLM (± item SE)")

        ax.set_title(panel_name)
        ax.set_xlabel("Sentence type")
        ax.set_xticks([xmap[t] for t in types], labels=types)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.5, len(types) + 0.5)
        ax.grid(alpha=0.25)

        if j == 0:
            ax.set_ylabel("Scaled measure (min–max normalized)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    hH, hL = axes[0].get_legend_handles_labels()
    if hH:
        axes[0].legend(loc="best", frameon=True, fontsize=8)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure.png")

def main():
    human_long = load_human_means(human_means_csv)
    type_categories = list(human_long["type"].cat.categories)
    llm_sum = load_llm_bits_summary(llm_items_csv, type_categories=type_categories)

    hum_sc, llm_sc = scale_within_experiment(human_long, llm_sum)
    plot_exp2_scaled(hum_sc, llm_sc)

if __name__ == "__main__":
    main()

