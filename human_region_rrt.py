import pandas as pd
import re

# file paths
human_file = "Data/human_raw.csv"
region_file = "Data/region_map_exp1.csv"
output_file = "Data/human_region_rt.csv"

#parse the regions of interest
def parse_region(spec):
    if pd.isna(spec): return []
    s = str(spec).strip().strip('"').strip("'")
    s = s.replace("–","-").replace("—","-").replace("−","-")
    idxs = []
    for part in re.split(r"[,\s]+", s):
        if not part: continue
        if "-" in part:
            a,b = part.split("-",1); a,b = int(a), int(b)
            if a>b: a,b = b,a
            idxs.extend(range(a,b+1))
        else:
            idxs.append(int(part))
    return idxs

def main():
    #load the human df
    human = pd.read_csv(human_file, sep=None, engine="python", encoding="utf-8-sig")
    human.columns = human.columns.str.strip().str.lower()

    # load the region maps
    regions = pd.read_csv(region_file, sep=None, engine="python", encoding="utf-8-sig")
    regions.columns = regions.columns.str.strip().str.lower()

    #   normalize the text in columns
    if "type" in regions.columns:
        regions["type"] = (
            regions["type"].astype(str)
            .str.replace(r"[–—−]", "-", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    #define the necessasry human columns
    needed_h = {"id","item","cond"}
    missing = [c for c in needed_h if c not in human.columns]

   #make sure the labels match 
    cond_to_type = {
        "sr":   "Subject",
        "tor":  "Object-Topicalized",
        "orsi": "Object-Subject-Inversion",
    }

    human["type"] = human["cond"].astype(str).str.strip().str.lower().map(cond_to_type)

    #get the region columns
    region_cols = [c for c in regions.columns if c.startswith("region") and not c.endswith("_surprisal")]

    #Merge the dfs 
    join_cols = ["item","type"] + region_cols
    reg_small = regions[join_cols].copy()
    merged = human.merge(reg_small, on=["item","type"], how="inner")

    #get readint time by region column
    for rc in region_cols:
        outc = f"{rc}_rrt"
        def _sum_region(row):
            idxs = parse_region(row[rc])
            vals = []
            for i in idxs:
                col = f"w{i}"
                if col in merged.columns:
                    v = row[col]
                    if pd.notna(v): vals.append(float(v))
            return sum(vals) if vals else None
        merged[outc] = merged.apply(_sum_region, axis=1)

    #save per participant
    out_cols = ["id","item","type","cond"] + [f"{rc}_rrt" for rc in region_cols]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(output_file, index=False, encoding="utf-8")
    print("saved individual rts")

   
   
   #save a summary
    region_rrt_cols = [f"{rc}_rrt" for rc in region_cols]
    by_type = merged.groupby("type", dropna=False)[region_rrt_cols].mean().reset_index()
    by_type.to_csv(output_file.replace(".csv", ".means_by_clause_type.csv"), index=False, encoding="utf-8")
    print("Saved averages by type.")

    if "item" in merged.columns:
        by_item = merged.groupby(["item","type"], dropna=False)[region_rrt_cols].mean().reset_index()
        by_item.to_csv(output_file.replace(".csv", ".means_by_item.csv"), index=False, encoding="utf-8")
        print("Saved averages by item by type")

if __name__ == "__main__":
    main()

