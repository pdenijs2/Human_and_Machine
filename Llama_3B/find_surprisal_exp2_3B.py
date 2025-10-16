import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name and authentication token
model_name = "meta-llama/Llama-3.2-3B"
hf_token = os.environ.get("HF_TOKEN")

# File paths
input_file = "./Data/region_map_exp2.csv"  
output_dir = "./Data"  
output_prefix = "Experiment_2_3B_"  

# Setting the number of CPU threads
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

# handling GPU setup if available
#changed to ieee
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "ieee"  
    torch.backends.cudnn.conv_fp32_precision = "ieee" 
    print("Using GPU")

# Loading the tokenizer from guggingface 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

# Loading the model GPU if possible
model = None
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        torch_dtype=torch.float32,
        device_map="auto", 
        low_cpu_mem_usage=True,
    )
    print("Model loaded on GPU.")
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    print("Model loaded on CPU.")

model.eval()


# Calculate word-level surprisal
def calc_word_surprisal(text, model, tokenizer):
    # tokenize
    encoded = tokenizer(str(text), return_tensors="pt")
    
    device = next(model.parameters()).device
    
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits

    input_ids = encoded["input_ids"]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    word_surps = []
    curr = 0.0

    # Loop through the tokens and calculate surprisal
    for i in range(1, input_ids.size(1)):
        target_id = input_ids[0, i]
        
        #log probabilities
        log_probs = torch.log_softmax(logits[0, i - 1], dim=-1)
        
        #surprisal = negative log likelihood
        
        tok_surp = -log_probs[target_id].item()

        tok = tokens[i]

        # stop at the periods and end of sentences
        if tok.strip() in {'.', '_.', 'Ġ.', '</s>'}:  
            break

        # Check if this token is the start of a new word
        new_word = (i == 1) or tok.startswith("▁") or tok.startswith("Ġ")
        
        
        
        if new_word:
            if curr > 0:
                word_surps.append(curr)
            curr = tok_surp
        else:
            curr += tok_surp

    if curr > 0:
    
        word_surps.append(curr)

    return word_surps


#  Function to parse  region information
def parse_regions(spec):
    # If Nan return an empty list
    if pd.isna(spec):
        return []
    
    spec = str(spec).strip().strip('"\'')
    if not spec:
        return []
    
    #  fix the mixed dashed
    spec = spec.replace("–", "-").replace("—", "-")

    # Geet the regions split
    idxs = []
    for part in spec.replace(",", " ").split():
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            if start > end:
                start, end = end, start
            idxs.extend(range(start, end + 1))
        
        else:
            idxs.append(int(part))

    return idxs


# main function to process the data and save results = word level, by type, by item x type
def main():
    df = pd.read_csv(input_file)
    
    # Get word-level surprisal for all sentences
    all_surps = []
    
    #remember to add tqdm
    for sent in tqdm(df["Text"]):
        surps = calc_word_surprisal(sent, model, tokenizer)
        all_surps.append(surps)

    # Add word-level surprisal columns 
    max_words = max(len(s) for s in all_surps)
    for i in range(max_words):
    
        df[f"W{i+1}"] = [
            round(s[i], 6) if i < len(s) else None
            for s in all_surps
        ]

    # find all region colunms (
    region_cols = [c for c in df.columns if c.lower().startswith("region") and not c.endswith("_Surprisal")]
    
    # find surprisal for each region
    for rc in region_cols:
        def sum_region(row):
            idxs = parse_regions(row[rc])
            if not idxs: 
                return None
            
            total = 0.0 
            for idx in idxs:
                col = f"W{idx}"
                if col in df.columns and pd.notna(row.get(col)):
                    total += row[col]
            
            return round(total, 6) if total > 0 else None  
        
        df[f"{rc}_Surprisal"] = df.apply(sum_region, axis=1)

    # save the final file
    outfile = os.path.join(output_dir, f"{output_prefix}_word_and_region_surprisal.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved to {outfile}")

    # Calculate and save average surprisals by type
    if "Type" in df.columns and region_cols:
        reg_cols = [f"{rc}_Surprisal" for rc in region_cols]
        means = df.groupby("Type")[reg_cols].mean().reset_index()
        mfile = os.path.join(output_dir, f"{output_prefix}_region_avg_by_type.csv")
        means.to_csv(mfile, index=False)
        print(f"saved to {mfile}")

    # calculate and save average surprisals by item by type
    if "Item" in df.columns and "Type" in df.columns:
        by_item = df.groupby(["Item", "Type"])[reg_cols].mean().reset_index()
        ifile = os.path.join(output_dir, f"{output_prefix}_region_by_item.csv")
        by_item.to_csv(ifile, index=False)
        print(f"Saved to {ifile}")


if __name__ == "__main__":
    main()

