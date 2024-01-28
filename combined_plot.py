import torch
from torch import nn
import numpy as np
import cv2
import pickle
import os
import io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from PIL import Image

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
    "rnd_Latn": "Random punctuation [Latn]",
}

suffix_dict = {
    "mlp": "mlp",
    "fc2": "fc2",
    "attn": "attn.o_proj",
}

output_dict = {
    "opt_tensors/": "OPT 6.7B",
    "llama_tensors/": "LLaMa-2 7B",
    "mistral_tensors/": "Mistral 7B",
}

cmap = matplotlib.cm.get_cmap("cet_diverging_bwr_20_95_c54")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.labelsize"] = 8


def main():
    folders = ["opt_tensors", "llama_tensors", "mistral_tensors/"]
    output_folder = "combined_outputs"

    for folder in enumerate(folders):
        # Read the .jpg files in pairs of two and combine them (plot them one on top of the other)
        files = language_dict.keys()
        suffixes = suffix_dict.keys()

        for file_name in enumerate(files):
            if file_name != "eng_Latn":
                continue

        for i, file_name in enumerate(files):
            print(f"{i}/{len(files)}")
            suffix = None
            # Read img with cv2
            tensor1 = torch.load(folder + file_name + "_attn.pt")
            if "opt" in folder:
                tensor2 = torch.load(folder + file_name + "_fc2.pt")
                suffix = "FC2"
            else:
                tensor2 = torch.load(folder + file_name + "_mlp.pt")
                suffix = "MLP"

            plt.clf()
            # Create two plots
            fig, axs = plt.subplots(2, gridspec_kw={"wspace": 0, "hspace": 0})
            # fig.suptitle(f"{language_dict[file_name]}")
            tensors = [tensor1, tensor2]

            for idx, tensor in enumerate(tensors):
                activations = tensor.clone()
                # Find the distribution of activations
                activation_values = activations.view(-1).detach().numpy()
                activation_values_sorted = sorted(activation_values)

                # Calculate the thresholds for the top and bottom 1 percentile
                percentile_1 = int(0.01 * len(activation_values))
                threshold_low = activation_values_sorted[percentile_1]
                threshold_high = activation_values_sorted[-percentile_1]

                # Set activations to 0 that are not in the top and bottom 1 percentile
                activations = torch.where(
                    (activations >= threshold_low) & (activations <= threshold_high),
                    torch.tensor(0.0),
                    activations,
                )

                activations = nn.MaxPool1d(32)(activations) - nn.MaxPool1d(32)(
                    -activations
                )
                # activations *= 10

                activations = torch.log(torch.abs(activations) + 1) * torch.sign(
                    activations
                )

                axs[idx].imshow(
                    activations.squeeze(1).numpy(),
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=-3,
                    vmax=+3,
                )
                max_value = torch.max(tensor)
                min_value = torch.min(tensor)
                # Include the threshold values in the plot, as well as the highest and lowest activation values
                name = suffix if idx == 1 else "Attn o_proj"
                axs[idx].text(
                    115,
                    7.5,
                    f"Low 1%: {threshold_low:.2f}\nTop 1%: {threshold_high:.2f}\nMin: {min_value:.2f}\nMax: {max_value:.2f}",
                    fontsize=5,
                )
                axs[idx].text(
                    0,
                    2,
                    f"{name}",
                    fontsize=8,
                )
                axs[idx].tick_params(axis="both", which="major", labelsize=7)
                axs[idx].set_aspect("equal")

            axs[0].set_title(f"{output_dict[folder]} - {language_dict[file_name]}")
            axs[0].set(ylabel=f"Layer")
            axs[1].set(ylabel="Layer", xlabel="Hidden Dimension")

            x_ticks_displayed = [0, 16, 32, 48, 64, 80, 96, 112, 128]
            x_ticks_actual = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]

            axs[0].set_xticks(x_ticks_displayed, x_ticks_actual)
            axs[1].set_xticks(x_ticks_displayed, x_ticks_actual)

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()

            canvas = fig.canvas
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img_flat.reshape(int(height), int(width), 3)

            # Remove from img the rows from 482 to 700 using indexing numpy
            img = np.delete(img, np.s_[1020:1286], axis=0)
            img = np.delete(img, np.s_[:110], axis=0)
            img = np.delete(img, np.s_[-110:], axis=0)
            img = np.delete(img, np.s_[:70], axis=1)
            img = np.delete(img, np.s_[-60:], axis=1)

            im = Image.fromarray(img)
            im.save(f"{file_name}_{output_dict[folder]}.jpg")

            plt.close()


if __name__ == "__main__":
    main()
