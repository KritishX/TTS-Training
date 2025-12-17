# ===== COMPLETE WORKING VITS TRAINING SCRIPT =====
# This fixes all the errors in your notebook

# Define base directories
import os
BASE = r"C:\Users\ReticleX\Pictures\nepali_tts"  # Update this to your actual path
OUTPUT = os.path.join(BASE, "vits_output")


import sys
import torch
import json
import time
from pathlib import Path

print("=" * 70)
print("VITS TTS TRAINING - COMPLETE WORKING VERSION")
print("=" * 70)

# ===== STEP 2: IMPORTS =====
print("\n📦 Importing modules...")
try:
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.tts.utils.text.characters import Graphemes
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Install TTS: pip install TTS")
    sys.exit(1)

# ===== STEP 3: CREATE NEPALI CHARACTER SET =====
print("\n📝 Creating Nepali character set...")

nepali_vocab = []

# Vowels
vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ']
nepali_vocab.extend(vowels)

# Consonants
consonants = [
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह'
]
nepali_vocab.extend(consonants)

# Vowel signs
vowel_signs = ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्']
nepali_vocab.extend(vowel_signs)

# Diacritics
diacritics = ['ं', 'ः', 'ँ']
nepali_vocab.extend(diacritics)

# Nepali digits
digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
nepali_vocab.extend(digits)

# Latin alphabet and numbers
latin = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
nepali_vocab.extend(latin)

# Common punctuation
common_punct = list(" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।")
nepali_vocab.extend(common_punct)

# Remove duplicates and sort
nepali_vocab = sorted(set(nepali_vocab))

print(f"✅ Character set ready ({len(nepali_vocab)} characters)")

# Create Graphemes object
chars_obj = Graphemes(
    characters=nepali_vocab,
    punctuations="।!?,.:; -\"",
    pad="_",
    eos="~",
    bos="^",
    blank="#",
)

# Create tokenizer
tokenizer = TTSTokenizer(
    use_phonemes=False,
    characters=chars_obj,
    add_blank=True,
)

print(f"✅ Tokenizer ready (vocab: {len(tokenizer.characters.characters)})")

# Test tokenizer
test_text = "नमस्ते"
test_ids = tokenizer.text_to_ids(test_text)
test_decoded = tokenizer.ids_to_text(test_ids)
print(f"   Test: '{test_text}' → {len(test_ids)} tokens → '{test_decoded}'")

# ===== STEP 4: DATASET CONFIGURATION =====
print("\n📊 Setting up dataset...")

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train=os.path.join(BASE, "dataset", "ljspeech_train", "metadata.csv"),
    meta_file_val=os.path.join(BASE, "dataset", "ljspeech_val", "metadata.csv"),
    path=os.path.join(BASE, "dataset", "ljspeech_train"),
    language="ne",
)
# ===== STEP 6: LOAD DATASET SAMPLES =====
print("\n📂 Loading dataset samples...")

try:
    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.15,
    )
    
    print(f"✅ Data loaded:")
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Validation samples: {len(eval_samples)}")
    
    if len(train_samples) == 0:
        raise Exception("No training samples found!")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("\n💡 Check:")
    print(f"   1. File exists: {dataset_config.meta_file_train}")
    print(f"   2. Audio files exist in: {dataset_config.path}")
    print(f"   3. Format: filename|text (LJSpeech format)")
    sys.exit(1)

# ===== CRITICAL FIX: Use BaseAudioConfig instead of dict =====
# Replace your audio config creation with this:

from TTS.config.shared_configs import BaseAudioConfig

print("\n🎵 Creating audio config (FIXED)...")

# CORRECT: Use BaseAudioConfig object, not dict
audio_config = BaseAudioConfig(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    num_mels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,
)

# Create audio processor from config
ap = AudioProcessor.init_from_config(audio_config)
print(f"✅ Audio processor: {ap.sample_rate} Hz")

# Now when you create VitsConfig:
config = VitsConfig()
config.audio = audio_config  # This is now a proper object, not a dict!
config.output_path = OUTPUT
config.run_name = "nepali_vits"

# Set other attributes
config.datasets = [dataset_config]
config.batch_size = 4
config.eval_batch_size = 2
config.num_loader_workers = 0
config.num_eval_loader_workers = 0
config.epochs = 100
config.text_cleaner = "basic_cleaners"
config.use_phonemes = False
config.add_blank = True
config.characters = None
config.num_chars = len(tokenizer.characters.characters)
config.lr = 2e-4
config.print_step = 25
config.save_step = 1000
config.save_n_checkpoints = 5
config.run_eval = True
config.test_sentences = ["नमस्ते", "धन्यवाद"]

print("✅ Config created with proper audio config")

# Create model
model = Vits(
    config=config,
    ap=ap,
    tokenizer=tokenizer,
    speaker_manager=None,
)

if torch.cuda.is_available():
    model.cuda()

print(f"✅ Model ready ({sum(p.numel() for p in model.parameters()):,} params)")

# Create trainer
trainer_args = TrainerArgs()

trainer = Trainer(
    trainer_args,
    config,
    OUTPUT,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

print("✅ Trainer ready!")
print("\n🚀 Run: trainer.fit()")

# ========== CELL 7: VITS Configuration (FIXED) ==========
print("\n⚙️ VITS configuration...")

# Step 1: Create base config
config = VitsConfig(
    output_path=OUTPUT,
    run_name="nepali_vits",
)

# Step 2: Set attributes
config.datasets = [dataset_config]
config.audio = audio_config

config.batch_size = 4
config.eval_batch_size = 2
config.num_loader_workers = 0
config.num_eval_loader_workers = 0
config.epochs = 100

config.text_cleaner = "basic_cleaners"
config.use_phonemes = False
config.add_blank = True
config.characters = None
config.num_chars = len(tokenizer.characters.characters)

config.optimizer = "AdamW"
config.optimizer_params = {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01}
config.lr = 2e-4
config.lr_scheduler = "ExponentialLR"
config.lr_scheduler_params = {"gamma": 0.999875}

config.print_step = 50
config.plot_step = 0
config.dashboard_logger = None
config.save_step = 1000
config.save_n_checkpoints = 5
config.run_eval = True

config.test_sentences = ["नमस्ते", "धन्यवाद"]

print("✅ Config created")

print("=" * 70)
print("PART 1: TRAINING")
print("=" * 70)

print("\n🎬 Starting training...")
print(f"   Training samples: {len(train_samples)}")
print(f"   Validation samples: {len(eval_samples)}")
print(f"   Epochs: {config.epochs}")
print(f"   Batch size: {config.batch_size}")
print(f"   Output: {OUTPUT}")

print("\n" + "-" * 70)

try:
    trainer.fit()

    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    trainer.save_checkpoint()

except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
