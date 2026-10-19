# predict_c0c1f2.py
import csv
import json
import os
from datetime import datetime
import torch

from aptatrans_pipeline import AptaTransPipeline

# 1. Definisi sekuens asam amino target (c0c1f)
c0c1f_sequence = (
    "MPEPGKKPVSAFSKKPRSVEVAAGSPAVFEAETERAGVKVRWQRGGSDISASNKYGLATERGTRHTLTVREASADDQASYAAMDGTAKSEAYLVLREAEAAPAAELGESAPSPKGSSSAALNGPTPGAPDDPIGLFVMRPQDGEVTVGGSITFSARVAGASLLKPPVVKWFKGKWVDLSSKVGQHLQLHDSYDRASKVYLFELHITDAQPAFTGSYRCEVSTKDKFDCSNFNLTVHEAMGTGDLDLLSAFRRTSLAGGGRRISDSHED"
)

def save_session_checkpoint(pipeline, checkpoint_dir, session_name):
    """Menyimpan state dictionary model pipeline."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{session_name}_model.pt")
    
    if hasattr(pipeline, "model"):
        torch.save(pipeline.model.state_dict(), checkpoint_path)
    elif hasattr(pipeline, "save_model"):
        pipeline.save_model(checkpoint_path)
    else:
        torch.save(pipeline.__dict__, checkpoint_path)
        
    print(f"Checkpoint model tersimpan di: {checkpoint_path}")

def save_session_results(results, output_dir, session_name, target_seq):
    """Menyimpan output kandidat aptamer ke JSON dan CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simpan format JSON
    json_path = os.path.join(output_dir, f"{session_name}_results.json")
    session_data = {
        "session_id": session_name,
        "target_sequence": target_seq,
        "results": results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4)
        
    # 2. Simpan format CSV
    csv_path = os.path.join(output_dir, f"{session_name}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Generated_Aptamer_Sequence"])
        if isinstance(results, (list, tuple)):
            for idx, item in enumerate(results, 1):
                writer.writerow([idx, item])
        else:
            writer.writerow([1, results])

    print(f"Hasil rekomendasi tersimpan di:\n - {json_path}\n - {csv_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Perangkat: {device}")

    # Definisi konfigurasi sesi
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    # 2. Inisialisasi Pipeline AptaTrans
    pipeline = AptaTransPipeline(
        dim=128,
        mult_ff=2,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        load_best_pt=False,
        device=device,
        seed=1004
    )

    # 3. Pre-train Aptamer Encoder
    print("\n[1/4] Memuat data RNA & Pre-training Aptamer Encoder...")
    pipeline.set_data_rna_pt(batch_size=32)
    pipeline.pretrain_encoder_aptamer(epochs=10, lr=1e-5)

    # 4. Pre-train Protein Encoder
    print("\n[2/4] Memuat data Protein & Pre-training Protein Encoder...")
    pipeline.set_data_protein_pt(batch_size=32)
    pipeline.pretrain_encoder_protein(epochs=10, lr=1e-5)

    # 5. Fine-tuning pada Dataset Interaksi
    print("\n[3/4] Fine-tuning model pada dataset interaksi aptamer-protein...")
    pipeline.set_data_for_training(batch_size=16)
    pipeline.train(epochs=200, lr=1e-5)

    # Simpan bobot setelah training selesai
    save_session_checkpoint(pipeline, checkpoint_dir, session_id)

    # 6. Jalankan Generasi Rekomendasi Aptamer untuk c0c1f via Apta-MCTS
    print("\n[4/4] Merekomendasikan sekuens kandidat aptamer untuk c0c1f...")
    recommendations = pipeline.recommend(
        target=c0c1f_sequence, 
        n_aptamers=5, 
        depth=40, 
        iteration=1000
    )

    # Simpan hasil sekuens aptamer
    save_session_results(recommendations, results_dir, session_id, c0c1f_sequence)
    print("\nSesi selesai dan semua artefak telah diamankan.")

if __name__ == "__main__":
    main()