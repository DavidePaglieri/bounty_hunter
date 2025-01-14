import argparse
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import holoviews as hv

hv.extension("matplotlib")

language_dict = {
    "ace_Arab": "Acehnese (Arabic script) [Arab]",
    "ace_Latn": "Acehnese (Latin script) [Latn]",
    "acm_Arab": "Mesopotamian Arabic [Arab]",
    "acq_Arab": "Ta'izzi-Adeni Arabic [Arab]",
    "aeb_Arab": "Tunisian Arabic [Arab]",
    "afr_Latn": "Afrikaans [Latn]",
    "ajp_Arab": "South Levantine Arabic [Arab]",
    "aka_Latn": "Akan [Latn]",
    "amh_Ethi": "Amharic [Ethi]",
    "apc_Arab": "North Levantine Arabic [Arab]",
    "arb_Arab": "Modern Standard Arabic [Arab]",
    "arb_Latn": "Modern Standard Arabic (Romanized) [Latn]",
    "ars_Arab": "Najdi Arabic [Arab]",
    "ary_Arab": "Moroccan Arabic [Arab]",
    "arz_Arab": "Egyptian Arabic [Arab]",
    "asm_Beng": "Assamese [Beng]",
    "ast_Latn": "Asturian [Latn]",
    "awa_Deva": "Awadhi [Deva]",
    "ayr_Latn": "Central Aymara [Latn]",
    "azb_Arab": "South Azerbaijani [Arab]",
    "azj_Latn": "North Azerbaijani [Latn]",
    "bak_Cyrl": "Bashkir [Cyrl]",
    "bam_Latn": "Bambara [Latn]",
    "ban_Latn": "Balinese [Latn]",
    "bel_Cyrl": "Belarusian [Cyrl]",
    "bem_Latn": "Bemba [Latn]",
    "ben_Beng": "Bengali [Beng]",
    "bho_Deva": "Bhojpuri [Deva]",
    "bjn_Arab": "Banjar (Arabic script) [Arab]",
    "bjn_Latn": "Banjar (Latin script) [Latn]",
    "bod_Tibt": "Standard Tibetan [Tibt]",
    "bos_Latn": "Bosnian [Latn]",
    "bug_Latn": "Buginese [Latn]",
    "bul_Cyrl": "Bulgarian [Cyrl]",
    "cat_Latn": "Catalan [Latn]",
    "ceb_Latn": "Cebuano [Latn]",
    "ces_Latn": "Czech [Latn]",
    "cjk_Latn": "Chokwe [Latn]",
    "ckb_Arab": "Central Kurdish [Arab]",
    "cmn_Hans": "Mandarin Chinese (Simplified) [Hans]",
    "cmn_Hant": "Mandarin Chinese (Traditional) [Hant]",
    "crh_Latn": "Crimean Tatar [Latn]",
    "cym_Latn": "Welsh [Latn]",
    "dan_Latn": "Danish [Latn]",
    "deu_Latn": "German [Latn]",
    "dik_Latn": "Southwestern Dinka [Latn]",
    "dyu_Latn": "Dyula [Latn]",
    "dzo_Tibt": "Dzongkha [Tibt]",
    "ell_Grek": "Greek [Grek]",
    "eng_Latn": "English [Latn]",
    "epo_Latn": "Esperanto [Latn]",
    "est_Latn": "Estonian [Latn]",
    "eus_Latn": "Basque [Latn]",
    "ewe_Latn": "Ewe [Latn]",
    "fao_Latn": "Faroese [Latn]",
    "fij_Latn": "Fijian [Latn]",
    "fin_Latn": "Finnish [Latn]",
    "fon_Latn": "Fon [Latn]",
    "fra_Latn": "French [Latn]",
    "fur_Latn": "Friulian [Latn]",
    "fuv_Latn": "Nigerian Fulfulde [Latn]",
    "gla_Latn": "Scottish Gaelic [Latn]",
    "gle_Latn": "Irish [Latn]",
    "glg_Latn": "Galician [Latn]",
    "grn_Latn": "Guarani [Latn]",
    "guj_Gujr": "Gujarati [Gujr]",
    "hat_Latn": "Haitian Creole [Latn]",
    "hau_Latn": "Hausa [Latn]",
    "heb_Hebr": "Hebrew [Hebr]",
    "hin_Deva": "Hindi [Deva]",
    "hne_Deva": "Chhattisgarhi [Deva]",
    "hrv_Latn": "Croatian [Latn]",
    "hun_Latn": "Hungarian [Latn]",
    "hye_Armn": "Armenian [Armn]",
    "ibo_Latn": "Igbo [Latn]",
    "ilo_Latn": "Ilocano [Latn]",
    "ind_Latn": "Indonesian [Latn]",
    "isl_Latn": "Icelandic [Latn]",
    "ita_Latn": "Italian [Latn]",
    "jav_Latn": "Javanese [Latn]",
    "jpn_Jpan": "Japanese [Jpan]",
    "kab_Latn": "Kabyle [Latn]",
    "kac_Latn": "Jingpho [Latn]",
    "kam_Latn": "Kamba [Latn]",
    "kan_Knda": "Kannada [Knda]",
    "kas_Arab": "Kashmiri (Arabic script) [Arab]",
    "kas_Deva": "Kashmiri (Devanagari script) [Deva]",
    "kat_Geor": "Georgian [Geor]",
    "knc_Arab": "Central Kanuri (Arabic script) [Arab]",
    "knc_Latn": "Central Kanuri (Latin script) [Latn]",
    "kaz_Cyrl": "Kazakh [Cyrl]",
    "kbp_Latn": "Kabiyè [Latn]",
    "kea_Latn": "Kabuverdianu [Latn]",
    "khm_Khmr": "Khmer [Khmr]",
    "kik_Latn": "Kikuyu [Latn]",
    "kin_Latn": "Kinyarwanda [Latn]",
    "kir_Cyrl": "Kyrgyz [Cyrl]",
    "kmb_Latn": "Kimbundu [Latn]",
    "kmr_Latn": "Northern Kurdish [Latn]",
    "kon_Latn": "Kikongo [Latn]",
    "kor_Hang": "Korean [Hang]",
    "lao_Laoo": "Lao [Laoo]",
    "lij_Latn": "Ligurian [Latn]",
    "fil_Latn": "Filipino [Latn]",
    "lim_Latn": "Limburgish [Latn]",
    "lin_Latn": "Lingala [Latn]",
    "lit_Latn": "Lithuanian [Latn]",
    "lmo_Latn": "Lombard [Latn]",
    "ltg_Latn": "Latgalian [Latn]",
    "ltz_Latn": "Luxembourgish [Latn]",
    "lua_Latn": "Luba-Kasai [Latn]",
    "lug_Latn": "Ganda [Latn]",
    "luo_Latn": "Luo [Latn]",
    "lus_Latn": "Mizo [Latn]",
    "lvs_Latn": "Standard Latvian [Latn]",
    "mag_Deva": "Magahi [Deva]",
    "mai_Deva": "Maithili [Deva]",
    "mal_Mlym": "Malayalam [Mlym]",
    "mar_Deva": "Marathi [Deva]",
    "min_Arab": "Minangkabau (Arabic script) [Arab]",
    "min_Latn": "Minangkabau (Latin script) [Latn]",
    "mkd_Cyrl": "Macedonian [Cyrl]",
    "plt_Latn": "Plateau Malagasy [Latn]",
    "mlt_Latn": "Maltese [Latn]",
    "mni_Beng": "Meitei (Bengali script) [Beng]",
    "khk_Cyrl": "Halh Mongolian [Cyrl]",
    "mos_Latn": "Mossi [Latn]",
    "mri_Latn": "Maori [Latn]",
    "mya_Mymr": "Burmese [Mymr]",
    "nld_Latn": "Dutch [Latn]",
    "nno_Latn": "Norwegian Nynorsk [Latn]",
    "nob_Latn": "Norwegian Bokmål [Latn]",
    "npi_Deva": "Nepali [Deva]",
    "nqo_Nkoo": "N'Ko [Nkoo]",
    "nso_Latn": "Northern Sotho [Latn]",
    "nus_Latn": "Nuer [Latn]",
    "nya_Latn": "Nyanja [Latn]",
    "oci_Latn": "Occitan [Latn]",
    "gaz_Latn": "West Central Oromo [Latn]",
    "ory_Orya": "Odia [Orya]",
    "pag_Latn": "Pangasinan [Latn]",
    "pan_Guru": "Eastern Panjabi [Guru]",
    "pap_Latn": "Papiamento [Latn]",
    "pes_Arab": "Western Persian [Arab]",
    "pol_Latn": "Polish [Latn]",
    "por_Latn": "Portuguese [Latn]",
    "prs_Arab": "Dari [Arab]",
    "pbt_Arab": "Southern Pashto [Arab]",
    "quy_Latn": "Ayacucho Quechua [Latn]",
    "ron_Latn": "Romanian [Latn]",
    "run_Latn": "Rundi [Latn]",
    "rus_Cyrl": "Russian [Cyrl]",
    "sag_Latn": "Sango [Latn]",
    "san_Deva": "Sanskrit [Deva]",
    "sat_Olck": "Santali [Olck]",
    "scn_Latn": "Sicilian [Latn]",
    "shn_Mymr": "Shan [Mymr]",
    "sin_Sinh": "Sinhala [Sinh]",
    "slk_Latn": "Slovak [Latn]",
    "slv_Latn": "Slovenian [Latn]",
    "smo_Latn": "Samoan [Latn]",
    "sna_Latn": "Shona [Latn]",
    "snd_Arab": "Sindhi (Arabic script) [Arab]",
    "som_Latn": "Somali [Latn]",
    "sot_Latn": "Southern Sotho [Latn]",
    "spa_Latn": "Spanish [Latn]",
    "als_Latn": "Tosk Albanian [Latn]",
    "srd_Latn": "Sardinian [Latn]",
    "srp_Cyrl": "Serbian [Cyrl]",
    "ssw_Latn": "Swati [Latn]",
    "sun_Latn": "Sundanese [Latn]",
    "swe_Latn": "Swedish [Latn]",
    "swh_Latn": "Swahili [Latn]",
    "szl_Latn": "Silesian [Latn]",
    "tam_Taml": "Tamil [Taml]",
    "tat_Cyrl": "Tatar [Cyrl]",
    "tel_Telu": "Telugu [Telu]",
    "tgk_Cyrl": "Tajik [Cyrl]",
    "tha_Thai": "Thai [Thai]",
    "tir_Ethi": "Tigrinya [Ethi]",
    "taq_Latn": "Tamasheq (Latin script) [Latn]",
    "taq_Tfng": "Tamasheq (Tifinagh script) [Tfng]",
    "tpi_Latn": "Tok Pisin [Latn]",
    "tsn_Latn": "Tswana [Latn]",
    "tso_Latn": "Tsonga [Latn]",
    "tuk_Latn": "Turkmen [Latn]",
    "tum_Latn": "Tumbuka [Latn]",
    "tur_Latn": "Turkish [Latn]",
    "twi_Latn": "Twi [Latn]",
    "uig_Arab": "Uyghur [Arab]",
    "ukr_Cyrl": "Ukrainian [Cyrl]",
    "umb_Latn": "Umbundu [Latn]",
    "urd_Arab": "Urdu [Arab]",
    "uzn_Latn": "Northern Uzbek [Latn]",
    "vec_Latn": "Venetian [Latn]",
    "vie_Latn": "Vietnamese [Latn]",
    "war_Latn": "Waray [Latn]",
    "wol_Latn": "Wolof [Latn]",
    "xho_Latn": "Xhosa [Latn]",
    "ydd_Hebr": "Eastern Yiddish [Hebr]",
    "yor_Latn": "Yoruba [Latn]",
    "yue_Hant": "Yue Chinese [Hant]",
    "zgh_Tfng": "Standard Moroccan Tamazight [Tfng]",
    "zsm_Latn": "Standard Malay [Latn]",
    "zul_Latn": "Zulu [Latn]",
}

suffix_dict = {
    "mlp": "mlp",
    "fc2": "fc2",
    "attn": "attn.o_proj",
}

suffix_dict = {
    "mlp": "MLP",
    "fc2": "FC2",
    "attn": "attn.o_proj",
}

output_dict = {
    "opt_tensors": "OPT 6.7B",
    "llama_tensors": "LLaMa-2 7B",
    "mistral_tensors": "Mistral 7B",
}

# Configure matplotlib
cmap = matplotlib.cm.get_cmap("cet_diverging_bwr_20_95_c54")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.labelsize"] = 8

colors = ["red", "blue", "green", "purple", "orange"]  # Extend if you have more folders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot histograms of .pt files from given folders."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more folders containing the .pt files (e.g. attn.pt, mlp.pt, fc2.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_folders = args.folders

    # Iterate through each folder separately
    for folder in input_folders:
        # Normalize folder path and skip if invalid
        folder = folder.rstrip("/")
        if not os.path.isdir(folder):
            print(f"Warning: '{folder}' is not a valid directory. Skipping.")
            continue

        # Prepare an output folder: original folder name + '_hist'
        out_folder = f"{folder}_hist"
        os.makedirs(out_folder, exist_ok=True)

        # Try to map folder name to a short label (or just use the folder name if not in dictionary)
        folder_base = os.path.basename(folder)
        legend_label = output_dict.get(folder_base, folder_base)

        # Gather all *.pt files in this folder
        pt_files = [
            f
            for f in os.listdir(folder)
            if f.endswith(".pt") and os.path.isfile(os.path.join(folder, f))
        ]

        # Dictionary to group files by (lang_code, suffix)
        # e.g. grouped_files[("eng_Latn", "attn")] = ["/path/to/eng_Latn_attn.pt", ...]
        grouped_files = {}

        for pt_file in pt_files:
            # Check which suffix it matches and parse language code
            if pt_file.endswith("_attn.pt"):
                suffix = "attn"
                lang_code = pt_file.replace("_attn.pt", "")
            elif pt_file.endswith("_mlp.pt"):
                suffix = "mlp"
                lang_code = pt_file.replace("_mlp.pt", "")
            elif pt_file.endswith("_fc2.pt"):
                suffix = "fc2"
                lang_code = pt_file.replace("_fc2.pt", "")
            else:
                # Not one of our recognized suffixes
                continue

            full_path = os.path.join(folder, pt_file)
            grouped_files.setdefault((lang_code, suffix), []).append(full_path)

        # For each (language_code, suffix) group, plot and save a figure
        for (lang_code, suffix), filepaths in grouped_files.items():
            # We’ll plot multiple histograms on the same figure only if
            # there are multiple .pt files that match the same (lang_code, suffix).
            # But often there's just one file per group.
            plt.figure()

            # Decide on histogram range, ticks, and suffix title
            if suffix == "attn":
                hist_range = (-0.75, 0.75)
                x_ticks = np.arange(-0.75, 1.0, 0.25)
                suffix_title = suffix_dict[suffix]
                x_label = "Value"
            else:
                # mlp or fc2
                hist_range = (-1.5, 1.5)
                x_ticks = np.arange(-1.5, 1.75, 0.5)
                suffix_title = suffix_dict[suffix]
                x_label = "Activation"

            # Plot each .pt file in this group
            for idx_, filepath in enumerate(filepaths):
                tensor = torch.load(filepath)
                # Flatten the tensor to 1D
                data = tensor.view(-1).detach().cpu().numpy()
                plt.hist(
                    data,
                    bins=1000,
                    alpha=0.5,
                    label=(
                        f"{legend_label}"
                        if len(filepaths) == 1
                        else f"{legend_label}_{idx_}"
                    ),
                    color=colors[idx_ % len(colors)],
                    range=hist_range,
                )

            # If available, translate the language code to a nicer name
            lang_title = language_dict.get(lang_code, lang_code)
            plt.title(f"{lang_title} - {suffix_title}")

            plt.xticks(x_ticks)
            plt.grid(
                True,
                which="both",
                axis="both",
                linestyle="--",
                alpha=0.5,
                linewidth=0.2,
            )
            plt.legend()
            plt.yscale("log")
            plt.xlabel(x_label)
            plt.ylabel("Frequency")

            # Save figure to the out_folder
            out_filename = f"{lang_code}_{suffix}_hist.png"
            out_path = os.path.join(out_folder, out_filename)
            plt.savefig(out_path)
            plt.close()

        print(f"Done processing folder '{folder}'. Results saved to '{out_folder}'.")


if __name__ == "__main__":
    main()
