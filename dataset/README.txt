TTS READY DATASET - LJSPEECH FORMAT
============================================================

DESCRIPTION:
--------------------
This dataset is formatted in LJSpeech style for TTS training.

STRUCTURE:
--------------------
dataset/
├── dataset_statistics.json
├── README.txt
├── ljspeech_train/
│   ├── wavs/           # 6082 files
│   └── metadata.csv
├── ljspeech_val/
│   ├── wavs/           # 1065 files
│   └── metadata.csv
├── ljspeech_test/
│   ├── wavs/           # 1229 files
│   └── metadata.csv

METADATA FORMAT:
--------------------
filename|text|text_normalized

STATISTICS:
--------------------
Total samples: 8376
Total duration: 9.76 hours

TRAIN:
  Samples: 6082
  Duration: 7.37 hours

VAL:
  Samples: 1065
  Duration: 1.07 hours

TEST:
  Samples: 1229
  Duration: 1.33 hours

