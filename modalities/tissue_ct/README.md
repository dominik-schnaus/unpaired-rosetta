# Tissue ↔ CT (CPTAC)

A diagnostic pathology slide and a CT scan of the same cancer patient, from the CPTAC collections of the NCI Imaging
Data Commons. Row `i` of both files is patient `i` of `patients.txt` (168 patients, sorted ids). `cohort.csv` lists the
309 candidates with one slide and one CT series.

| side | file | shape | encoder |
| --- | --- | --- | --- |
| pathology | `embeddings/CPTAC-path-ct/pathology/h_optimus_0.pt` | 168 x 1536, float32 | H-Optimus-0, mean over up to 196 tissue tiles at about 20x |
| ct | `embeddings/CPTAC-path-ct/ct/ct_fm.pt` | 168 x 512, float32 | CT-FM (SegResNet encoder), mean over its last feature map |

## Data

`download.py` fetches every patient's slide (DICOM-WSI) and CT series from the IDC, caches the tiles and a
`[96, 224, 224]` volume in `cache/<patient>.pt`, and deletes the DICOM again: 118.6 GB are downloaded, 4.5 GB stay
(measured on the folder that produced the stored embeddings). H-Optimus-0 is a gated model on the Hugging Face Hub.

## Commands

```sh
pixi run -e modalities python -m modalities.tissue_ct.download
pixi run -e modalities python -m modalities.tissue_ct.embed      # needs a GPU
pixi run -e modalities pytest tests/embeddings/test_tissue_ct.py -s
```

The test rebuilds the cache of one patient from its DICOM (equal) and embeds the first patients (cosine similarity
above 0.999).
