#!/usr/bin/env python3
"""
scripts/manifest_loader.py
==========================
Centralized Manifest Loader & Schema Parser for SANADA Suite
Member 5: Data Engineering & Backend API (Week 1 Deliverable)

Provides structured data access to:
- Member 2: Audio Source Count Lead (returns mixture audio + integer source count K)
- Member 3: Label Tagging Lead (returns mixture audio + multi-hot binary label vectors)
- Member 4: Audio Separation Lead (returns mixture audio + clean source stem paths)
- Member 1: Frontend Engine (supplies sample manifests for demo testing)
"""

import os
import ast
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SANADA_MIX_DIR = os.path.join(BASE_DIR, "SANADA", "Org_Fsd_Mix")

# 32 Curated FSD Environmental Classes defined in SANADA paper
FSD_32_CLASSES = sorted([
    "Acoustic_guitar", "Applause", "Bark", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Drawer_open_or_close", "Electric_piano",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Keys_jangling", "Laughter", "Meow", "Microwave_oven",
    "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak",
    "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"
])

ORG_4_SPEAKERS = ["Speaker_4_1", "Speaker_4_2", "Speaker_4_3", "Speaker_4_4"]
ORG_9_SPEAKERS = [f"Speaker_{i}" for i in range(1, 10)]

def safe_parse_list(val):
    """Safely parses JSON-formatted string lists from metadata.csv."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return [str(val).strip()]

class SANADAManifestLoader:
    def __init__(self, base_mix_dir=SANADA_MIX_DIR):
        self.base_mix_dir = base_mix_dir

    def get_manifest_path(self, dataset_name, split="train"):
        """Locates metadata.csv for a dataset and split (train/val/test)."""
        csv_path = os.path.join(self.base_mix_dir, dataset_name, split, "metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Manifest not found: {csv_path}")
        return csv_path

    def load_raw_manifest(self, dataset_name, split="train"):
        """Loads and parses the raw metadata.csv dataframe."""
        csv_path = self.get_manifest_path(dataset_name, split)
        df = pd.read_csv(csv_path)
        
        # Parse serialized list fields
        df["speaker_labels"] = df["speaker_labels"].apply(safe_parse_list)
        df["source_files"] = df["source_files"].apply(safe_parse_list)
        
        # Ensure mixture_path is absolute or project-relative
        df["mixture_abs_path"] = df["mixture_path"].apply(
            lambda p: os.path.abspath(os.path.join(BASE_DIR, p)) if not os.path.isabs(p) else p
        )
        return df

    def get_counting_data(self, dataset_name, split="train"):
        """Helper for Member 2: returns dataframe with audio path and target source count K."""
        df = self.load_raw_manifest(dataset_name, split)
        return df[["mixture_abs_path", "num_speakers", "duration_s", "sample_rate_hz"]]

    def get_tagging_data(self, dataset_name, split="train"):
        """Helper for Member 3: returns dataframe with audio path and multi-hot vector."""
        df = self.load_raw_manifest(dataset_name, split)
        
        # Determine taxonomy
        if "FSD" in dataset_name.upper():
            classes = FSD_32_CLASSES
        elif "9" in dataset_name:
            classes = ORG_9_SPEAKERS
        else:
            classes = ORG_4_SPEAKERS
            
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        def encode_labels(labels):
            vec = np.zeros(len(classes), dtype=np.float32)
            for lbl in labels:
                if lbl in class_to_idx:
                    vec[class_to_idx[lbl]] = 1.0
            return vec

        df["multi_hot"] = df["speaker_labels"].apply(encode_labels)
        df["classes_list"] = [classes] * len(df)
        return df[["mixture_abs_path", "speaker_labels", "multi_hot", "classes_list"]]

    def get_separation_data(self, dataset_name, split="train"):
        """Helper for Member 4: returns dataframe with mixture path, stem paths, and count."""
        df = self.load_raw_manifest(dataset_name, split)
        
        # Resolve source stem absolute paths
        def resolve_stems(stems):
            return [os.path.abspath(os.path.join(BASE_DIR, s)) if not os.path.isabs(s) else s for s in stems]
            
        df["source_stems_abs"] = df["source_files"].apply(resolve_stems)
        return df[["mixture_abs_path", "num_speakers", "source_stems_abs", "speaker_labels"]]

if __name__ == "__main__":
    loader = SANADAManifestLoader()
    print("Testing SANADA Manifest Loader (Member 5)...")
    
    # Test Org-Mix-4
    df_count = loader.get_counting_data("Org_Mix_4_Sounds_Variable", "train")
    print(f" Org-Mix-4 Counting Loaded: {len(df_count)} records. Sample K={df_count['num_speakers'].iloc[0]}")
    
    # Test FSD-Mix
    df_tag = loader.get_tagging_data("FSD_Mix_Variable_Sounds", "train")
    sample_tags = df_tag['speaker_labels'].iloc[0]
    sample_vec = df_tag['multi_hot'].iloc[0]
    print(f" FSD-Mix Tagging Loaded: {len(df_tag)} records. Sample Labels: {sample_tags}, Active indices: {np.where(sample_vec==1)[0]}")
    
    # Test Separation
    df_sep = loader.get_separation_data("Org_Mix_4_Sounds_Variable", "val")
    sample_stems = df_sep['source_stems_abs'].iloc[0]
    print(f" Separation Data Loaded: {len(df_sep)} val records. Stems to reconstruct: {len(sample_stems)}")
    print("Manifest loader verified successfully!")
