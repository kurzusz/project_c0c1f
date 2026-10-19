# predict_c0c1f.py

# 1. Definisi sekuens asam amino target (c0c1f)
c0c1f_sequence = "MPEPGKKPVSAFSKKPRSVEVAAGSPAVFEAETERAGVKVRWQRGGSDISASNKYGLATERGTRHTLTVREASADDQASYAAMDGTAKSEAYLVLREAEAAPAAELGESAPSPKGSSSAALNGPTPGAPDDPIGLFVMRPQDGEVTVGGSITFSARVAGASLLKPPVVKWFKGKWVDLSSKVGQHLQLHDSYDRASKVYLFELHITDAQPAFTGSYRCEVSTKDKFDCSNFNLTVHEAMGTGDLDLLSAFRRTSLAGGGRRISDSHED"

# 2. Contoh sekuens nukleotida aptamer awal
sample_aptamer = "CGTGCAGTACGCCAACCTTTCTCATGCGCTGCCCCTCTTA"

# 3. Verifikasi panjang sekuens menggunakan fungsi len() bawaan Python
print(f"Panjang Sekuens Protein c0c1f : {len(c0c1f_sequence)} Asam Amino")
print(f"Panjang Sekuens Aptamer Sampel : {len(sample_aptamer)} Nukleotida")

def extract_kmers(sequence, k=3):
     """
     Fungsi sederhana untuk memotong sekuens nukleotida menjadi token 3-mer.
     """
     kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1 )]
     return kmers

# Uji coba fungsi k-mers pada sampel aptamer
tokens = extract_kmers(sample_aptamer, k =3)
print("hasil Tokenisasi 3-mer Aptamer:", tokens)

def run_pipeline_preparation(target_protein_seq):
    print("\n--- Memulai Persiapan Prediksi Aptamer ---")

    # Validasi sederhana sekuens input
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    is_valid = all(char.upper() in valid_amino_acids for char in target_protein_seq)

    if is_valid:
        print("Sekuens protein valid!")
    else:
        print("Sekuens protein tidak valid.")

# Panggil fungsi dengan variabel c0c1f_sequence yang sudah didefinisikan sebelumnya
run_pipeline_preparation(c0c1f_sequence)

# Tempat di mana kode AptaTransPipeline akan dipanggil nanti:
# pipeline = AptaTransPipeline(dim=128, device='cuda')
# pipeline.recommend(target=target_protein_seq, n_aptamers=5)

# Titik masuk eksekusi utama (Main Entry Point)
if __name__ == "__main__":
    run_pipeline_preparation(coc1f_sequence)