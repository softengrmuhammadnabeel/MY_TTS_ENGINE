# TTS Models Reference Guide

## Text-to-Speech (TTS) Models

### Multilingual Models

| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **xtts_v2** | Multilingual | multi-dataset | Advanced multilingual TTS model supporting 17 languages with cross-language voice cloning capabilities. Latest version from Coqui AI with improved quality and stability. | None (End-to-end) |
| **xtts_v1.1** | Multilingual | multi-dataset | Multilingual TTS model supporting 14 languages with cross-language voice cloning. Includes reference leak fixes for better voice consistency. | None (End-to-end) |
| **your_tts** | Multilingual | multi-dataset | Research-grade multilingual TTS model accompanying the paper "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion". | None (End-to-end) |
| **bark** | Multilingual | multi-dataset | Generative audio model from Suno AI capable of producing highly realistic speech with emotions, music, and sound effects. | None (End-to-end) |

### English Models

#### LJSpeech Dataset
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2-DDC** | ljspeech | Tacotron2 with Double Decoder Consistency for improved speech quality and stability. | hifigan_v2 |
| **tacotron2-DDC_ph** | ljspeech | Tacotron2 with Double Decoder Consistency trained with phonemes for better pronunciation accuracy. | univnet |
| **glow-tts** | ljspeech | Flow-based TTS model offering fast parallel synthesis with good quality. | multiband-melgan |
| **speedy-speech** | ljspeech | Fast, non-autoregressive TTS model using Alignment Network for duration prediction. | hifigan_v2 |
| **tacotron2-DCA** | ljspeech | Tacotron2 with Decoder Consistency Architecture for enhanced speech synthesis. | multiband-melgan |
| **vits** | ljspeech | End-to-end TTS model combining variational inference with adversarial training for high-quality speech. | None (End-to-end) |
| **vits--neon** | ljspeech | Enhanced VITS model with optimizations for improved performance and quality. | None (End-to-end) |
| **fast_pitch** | ljspeech | FastPitch model using Aligner Network for fast, high-quality speech synthesis. | hifigan_v2 |
| **overflow** | ljspeech | Overflow model designed for robust and high-quality speech synthesis. | hifigan_v2 |
| **neural_hmm** | ljspeech | Neural Hidden Markov Model approach to TTS with probabilistic duration modeling. | hifigan_v2 |

#### VCTK Dataset (Multi-speaker)
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **vits** | vctk | Multi-speaker VITS model trained on 109 different English speakers with various accents. | None (End-to-end) |
| **fast_pitch** | vctk | Multi-speaker FastPitch model supporting various English speakers and accents. | None (End-to-end) |

#### Other English Datasets
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2** | ek1 | Tacotron2 model trained on EK1 dataset with received pronunciation (RP) English accent. | wavegrad |
| **tacotron-DDC** | sam | Tacotron2 with Double Decoder Consistency trained on Accenture's Sam dataset. | hifigan_v2 |
| **capacitron-t2-c50** | blizzard2013 | Tacotron2 enhanced with Capacitron attention mechanism (capacity=50) for improved long-form synthesis. | hifigan_v2 |
| **capacitron-t2-c150_v2** | blizzard2013 | Advanced Capacitron model with higher capacity (150) for better attention and longer sequences. | hifigan_v2 |
| **tortoise-v2** | multi-dataset | High-quality TTS model optimized for expressiveness and natural speech patterns. | None (End-to-end) |
| **jenny** | jenny | VITS model trained on Jenny (Dioco) dataset, providing clear female voice synthesis. | None (End-to-end) |

### European Languages

#### Spanish Models
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2-DDC** | mai | Spanish TTS model with Double Decoder Consistency for improved speech quality. | fullband-melgan |
| **vits** | css10 | End-to-end Spanish TTS model trained on CSS10 dataset for natural speech synthesis. | None (End-to-end) |

#### French Models
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2-DDC** | mai | French TTS model with Double Decoder Consistency architecture. | fullband-melgan |
| **vits** | css10 | End-to-end French TTS model providing natural-sounding speech synthesis. | None (End-to-end) |

#### German Models
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2-DCA** | thorsten | German TTS model with Decoder Consistency Architecture for high-quality synthesis. | fullband-melgan |
| **vits** | thorsten | End-to-end German TTS model trained on Thorsten dataset for natural speech. | None (End-to-end) |
| **tacotron2-DDC** | thorsten | German Tacotron2 model with Double Decoder Consistency (Dec2021 22k version). | hifigan_v1 |
| **vits-neon** | css10 | Enhanced German VITS model with optimizations for improved performance. | None (End-to-end) |

#### Italian Models
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **glow-tts** | mai_female | Italian female voice GlowTTS model for fast parallel synthesis. | None (End-to-end) |
| **vits** | mai_female | Italian female voice VITS model for high-quality speech synthesis. | None (End-to-end) |
| **glow-tts** | mai_male | Italian male voice GlowTTS model providing masculine voice characteristics. | None (End-to-end) |
| **vits** | mai_male | Italian male voice VITS model for natural male speech synthesis. | None (End-to-end) |

#### Dutch Models
| Model Name | Dataset | Description | Vocoder |
|------------|---------|-------------|---------|
| **tacotron2-DDC** | mai | Dutch TTS model with Double Decoder Consistency for improved speech quality. | parallel-wavegan |
| **vits** | css10 | End-to-end Dutch TTS model trained on CSS10 dataset for natural speech synthesis. | None (End-to-end) |

#### Other European Languages
| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **glow-tts** | Ukrainian | mai | Ukrainian GlowTTS model for fast parallel speech synthesis. | multiband-melgan |
| **vits** | Ukrainian | mai | Ukrainian VITS model for high-quality end-to-end speech synthesis. | None (End-to-end) |
| **vits** | Bulgarian | cv | Bulgarian TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Czech | cv | Czech TTS model trained on Common Voice dataset for natural speech. | None (End-to-end) |
| **vits** | Danish | cv | Danish TTS model providing natural-sounding speech synthesis. | None (End-to-end) |
| **vits** | Estonian | cv | Estonian TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Irish | cv | Irish Gaelic TTS model for natural speech synthesis. | None (End-to-end) |
| **vits** | Greek | cv | Greek TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Finnish | css10 | Finnish TTS model providing natural-sounding speech synthesis. | None (End-to-end) |
| **vits** | Croatian | cv | Croatian TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Lithuanian | cv | Lithuanian TTS model for natural speech synthesis. | None (End-to-end) |
| **vits** | Latvian | cv | Latvian TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Maltese | cv | Maltese TTS model providing natural-sounding speech synthesis. | None (End-to-end) |
| **vits** | Polish | mai_female | Polish female voice TTS model for natural speech synthesis. | None (End-to-end) |
| **vits** | Portuguese | cv | Portuguese TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Romanian | cv | Romanian TTS model for natural speech synthesis. | None (End-to-end) |
| **vits** | Slovak | cv | Slovak TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Slovenian | cv | Slovenian TTS model providing natural-sounding speech synthesis. | None (End-to-end) |
| **vits** | Swedish | cv | Swedish TTS model trained on Common Voice dataset. | None (End-to-end) |
| **vits** | Hungarian | css10 | Hungarian TTS model for natural speech synthesis. | None (End-to-end) |
| **glow-tts** | Belarusian | common-voice | Belarusian GlowTTS model created by @alex73 for fast synthesis. | hifigan |

### Asian Languages

| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **tacotron2-DDC-GST** | Chinese (Mandarin) | baker | Chinese TTS model with Global Style Tokens for expressive speech synthesis. | None (End-to-end) |
| **tacotron2-DDC** | Japanese | kokoro | Japanese TTS model with Double Decoder Consistency trained on Kokoro dataset. | hifigan_v1 |

### Middle Eastern & South Asian Languages

| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **glow-tts** | Persian/Farsi | custom | Persian female voice GlowTTS model for text-to-speech synthesis. | None (End-to-end) |
| **vits-male** | Bengali | custom | Bengali male voice VITS model for comprehensive Bangla TTS applications. | None (End-to-end) |
| **vits-female** | Bengali | custom | Bengali female voice VITS model for natural Bangla speech synthesis. | None (End-to-end) |
| **glow-tts** | Turkish | common-voice | Turkish GlowTTS model trained on Common Voice dataset. | hifigan |

### African Languages

| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **vits** | Ewe | openbible | Ewe language TTS model trained on biblical texts from Biblica. | None (End-to-end) |
| **vits** | Hausa | openbible | Hausa language TTS model trained on biblical audio and text data. | None (End-to-end) |
| **vits** | Lingala | openbible | Lingala language TTS model for Central African speech synthesis. | None (End-to-end) |
| **vits** | Twi (Akuapem) | openbible | Twi Akuapem dialect TTS model trained on biblical texts. | None (End-to-end) |
| **vits** | Twi (Asante) | openbible | Twi Asante dialect TTS model for Ghanaian speech synthesis. | None (End-to-end) |
| **vits** | Yoruba | openbible | Yoruba language TTS model trained on biblical audio and text data. | None (End-to-end) |

### Other Languages

| Model Name | Language | Dataset | Description | Vocoder |
|------------|----------|---------|-------------|---------|
| **vits** | Catalan | custom | Catalan TTS model trained on multiple datasets (Festcat, Google Catalan TTS, Common Voice) with 257 speakers and 138 hours of speech. | None (End-to-end) |

## Voice Conversion Models

| Model Name | Language | Dataset | Description |
|------------|----------|---------|-------------|
| **freevc24** | Multilingual | vctk | Advanced voice conversion model supporting multilingual voice cloning and conversion based on FreeVC architecture. |

## Vocoder Models

### Universal Vocoders
| Model Name | Dataset | Description |
|------------|---------|-------------|
| **wavegrad** | libri-tts | Universal neural vocoder based on WaveGrad architecture for high-quality audio synthesis. |
| **fullband-melgan** | libri-tts | Universal fullband MelGAN vocoder for high-fidelity audio generation across frequencies. |

### Language-Specific Vocoders

#### English Vocoders
| Model Name | Dataset | Description |
|------------|---------|-------------|
| **wavegrad** | ek1 | English RP (Received Pronunciation) WaveGrad vocoder by NMStoker. |
| **multiband-melgan** | ljspeech | Multi-band MelGAN vocoder optimized for LJSpeech dataset characteristics. |
| **hifigan_v2** | ljspeech | High-fidelity GAN vocoder v2 providing excellent audio quality for English speech. |
| **univnet** | ljspeech | Universal neural vocoder fine-tuned for TacotronDDC_ph spectrograms. |
| **hifigan_v2** | blizzard2013 | HiFiGAN v2 vocoder adapted for Blizzard 2013 dataset characteristics. |
| **hifigan_v2** | vctk | HiFiGAN v2 vocoder fine-tuned for multi-speaker VCTK dataset. |
| **hifigan_v2** | sam | HiFiGAN v2 vocoder optimized for SAM dataset characteristics. |

#### European Language Vocoders
| Model Name | Language | Dataset | Description |
|------------|----------|---------|-------------|
| **parallel-wavegan** | Dutch | mai | Parallel WaveGAN vocoder optimized for Dutch speech synthesis. |
| **wavegrad** | German | thorsten | German WaveGrad vocoder for Thorsten dataset. |
| **fullband-melgan** | German | thorsten | German fullband MelGAN vocoder for high-quality audio synthesis. |
| **hifigan_v1** | German | thorsten | HiFiGAN v1 vocoder for Thorsten Neutral Dec2021 22k sample rate model. |
| **multiband-melgan** | Ukrainian | mai | Multi-band MelGAN vocoder optimized for Ukrainian speech. |
| **hifigan** | Turkish | common-voice | HiFiGAN vocoder trained on Turkish Common Voice dataset. |
| **hifigan** | Belarusian | common-voice | Belarusian HiFiGAN vocoder created by @alex73. |

#### Asian Language Vocoders
| Model Name | Language | Dataset | Description |
|------------|----------|---------|-------------|
| **hifigan_v1** | Japanese | kokoro | HiFiGAN v1 vocoder trained for Japanese Kokoro dataset by @kaiidams. |

---

*Note: Models marked with "None (End-to-end)" are complete TTS systems that don't require separate vocoders. Models with specific vocoders listed require the corresponding vocoder for audio synthesis.*








## 🐸Coqui.ai News
- 📣 ⓍTTSv2 is here with 16 languages and better performance across the board.
- 📣 ⓍTTS fine-tuning code is out. Check the [example recipes](https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech).
- 📣 ⓍTTS can now stream with <200ms latency.
- 📣 ⓍTTS, our production TTS model that can speak 13 languages, is released [Blog Post](https://coqui.ai/blog/tts/open_xtts), [Demo](https://huggingface.co/spaces/coqui/xtts), [Docs](https://tts.readthedocs.io/en/dev/models/xtts.html)
- 📣 [🐶Bark](https://github.com/suno-ai/bark) is now available for inference with unconstrained voice cloning. [Docs](https://tts.readthedocs.io/en/dev/models/bark.html)
- 📣 You can use [~1100 Fairseq models](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) with 🐸TTS.
- 📣 🐸TTS now supports 🐢Tortoise with faster inference. [Docs](https://tts.readthedocs.io/en/dev/models/tortoise.html)

<div align="center">
<img src="https://static.scarf.sh/a.png?x-pxid=cf317fe7-2188-4721-bc01-124bb5d5dbb2" />

## <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>


**🐸TTS is a library for advanced Text-to-Speech generation.**

🚀 Pretrained models in +1100 languages.

🛠️ Tools for training new models and fine-tuning existing models in any language.

📚 Utilities for dataset analysis and curation.
______________________________________________________________________

[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)
[![License](<https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg>)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/TTS.svg)](https://badge.fury.io/py/TTS)
[![Covenant](https://camo.githubusercontent.com/7d620efaa3eac1c5b060ece5d6aacfcc8b81a74a04d05cd0398689c01c4463bb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6e7472696275746f72253230436f76656e616e742d76322e3025323061646f707465642d6666363962342e737667)](https://github.com/coqui-ai/TTS/blob/master/CODE_OF_CONDUCT.md)
[![Downloads](https://pepy.tech/badge/tts)](https://pepy.tech/project/tts)
[![DOI](https://zenodo.org/badge/265612440.svg)](https://zenodo.org/badge/latestdoi/265612440)

![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/aux_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/data_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/docker.yaml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/inference_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/style_check.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/text_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/tts_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/vocoder_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests0.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests1.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests2.yml/badge.svg)
[![Docs](<https://readthedocs.org/projects/tts/badge/?version=latest&style=plastic>)](https://tts.readthedocs.io/en/latest/)

</div>

______________________________________________________________________

## 💬 Where to ask questions
Please use our dedicated channels for questions and discussion. Help is much more valuable if it's shared publicly so that more people can benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker]                  |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker]                  |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]                    |
| 🗯 **General Discussion**       | [GitHub Discussions] or [Discord]   |

[github issue tracker]: https://github.com/coqui-ai/tts/issues
[github discussions]: https://github.com/coqui-ai/TTS/discussions
[discord]: https://discord.gg/5eXr5seRrv
[Tutorials and Examples]: https://github.com/coqui-ai/TTS/wiki/TTS-Notebooks-and-Tutorials


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 💼 **Documentation**              | [ReadTheDocs](https://tts.readthedocs.io/en/latest/)
| 💾 **Installation**               | [TTS/README.md](https://github.com/coqui-ai/TTS/tree/dev#installation)|
| 👩‍💻 **Contributing**               | [CONTRIBUTING.md](https://github.com/coqui-ai/TTS/blob/main/CONTRIBUTING.md)|
| 📌 **Road Map**                   | [Main Development Plans](https://github.com/coqui-ai/TTS/issues/378)
| 🚀 **Released Models**            | [TTS Releases](https://github.com/coqui-ai/TTS/releases) and [Experimental Models](https://github.com/coqui-ai/TTS/wiki/Experimental-Released-Models)|
| 📰 **Papers**                    | [TTS Papers](https://github.com/erogol/TTS-papers)|


## 🥇 TTS Performance
<p align="center"><img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/TTS-performance.png" width="800" /></p>

Underlined "TTS*" and "Judy*" are **internal** 🐸TTS models that are not released open-source. They are here to show the potential. Models prefixed with a dot (.Jofish .Abe and .Janice) are real human voices.

## Features
- High-performance Deep Learning models for Text2Speech tasks.
    - Text2Spec models (Tacotron, Tacotron2, Glow-TTS, SpeedySpeech).
    - Speaker Encoder to compute speaker embeddings efficiently.
    - Vocoder models (MelGAN, Multiband-MelGAN, GAN-TTS, ParallelWaveGAN, WaveGrad, WaveRNN)
- Fast and efficient model training.
- Detailed training logs on the terminal and Tensorboard.
- Support for Multi-speaker TTS.
- Efficient, flexible, lightweight but feature complete `Trainer API`.
- Released and ready-to-use models.
- Tools to curate Text2Speech datasets under```dataset_analysis```.
- Utilities to use and test your models.
- Modular (but not too much) code base enabling easy implementation of new ideas.

## Model Implementations
### Spectrogram models
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802)
- Align-TTS: [paper](https://arxiv.org/abs/2003.01950)
- FastPitch: [paper](https://arxiv.org/pdf/2006.06873.pdf)
- FastSpeech: [paper](https://arxiv.org/abs/1905.09263)
- FastSpeech2: [paper](https://arxiv.org/abs/2006.04558)
- SC-GlowTTS: [paper](https://arxiv.org/abs/2104.05557)
- Capacitron: [paper](https://arxiv.org/abs/1906.03402)
- OverFlow: [paper](https://arxiv.org/abs/2211.06892)
- Neural HMM TTS: [paper](https://arxiv.org/abs/2108.13320)
- Delightful TTS: [paper](https://arxiv.org/abs/2110.12612)

### End-to-End Models
- ⓍTTS: [blog](https://coqui.ai/blog/tts/open_xtts)
- VITS: [paper](https://arxiv.org/pdf/2106.06103)
- 🐸 YourTTS: [paper](https://arxiv.org/abs/2112.02418)
- 🐢 Tortoise: [orig. repo](https://github.com/neonbjb/tortoise-tts)
- 🐶 Bark: [orig. repo](https://github.com/suno-ai/bark)

### Attention Methods
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1910.10288)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)
- Dynamic Convolutional Attention: [paper](https://arxiv.org/pdf/1910.10288.pdf)
- Alignment Network: [paper](https://arxiv.org/abs/2108.10447)

### Speaker Encoder
- GE2E: [paper](https://arxiv.org/abs/1710.10467)
- Angular Loss: [paper](https://arxiv.org/pdf/2003.11982.pdf)

### Vocoders
- MelGAN: [paper](https://arxiv.org/abs/1910.06711)
- MultiBandMelGAN: [paper](https://arxiv.org/abs/2005.05106)
- ParallelWaveGAN: [paper](https://arxiv.org/abs/1910.11480)
- GAN-TTS discriminators: [paper](https://arxiv.org/abs/1909.11646)
- WaveRNN: [origin](https://github.com/fatchord/WaveRNN/)
- WaveGrad: [paper](https://arxiv.org/abs/2009.00713)
- HiFiGAN: [paper](https://arxiv.org/abs/2010.05646)
- UnivNet: [paper](https://arxiv.org/abs/2106.07889)

### Voice Conversion
- FreeVC: [paper](https://arxiv.org/abs/2210.15418)

You can also help us implement more models.

## Installation
🐸TTS is tested on Ubuntu 18.04 with **python >= 3.9, < 3.12.**.

If you are only interested in [synthesizing speech](https://tts.readthedocs.io/en/latest/inference.html) with the released 🐸TTS models, installing from PyPI is the easiest option.

```bash
pip install TTS
```

If you plan to code or train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/coqui-ai/TTS
pip install -e .[all,dev,notebooks]  # Select the relevant extras
```

If you are on Ubuntu (Debian), you can also run following commands for installation.

```bash
$ make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
$ make install
```

If you are on Windows, 👑@GuyPaddock wrote installation instructions [here](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system).


## Docker Image
You can also try TTS without install with the docker image.
Simply run the following command and you will be able to run TTS without installing it.

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

You can then enjoy the TTS server [here](http://[::1]:5002/)
More details about the docker images (like GPU support) can be found [here](https://tts.readthedocs.io/en/latest/docker_images.html)


## Synthesizing speech by 🐸TTS

### 🐍 Python API

#### Running a multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
```

#### Running a single speaker model

```python
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
```

#### Example voice conversion

Converting the voice in `source_wav` to the voice of `target_wav`

```python
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")
```

#### Example voice cloning together with the voice conversion model.
This way, you can clone voices by using any model in 🐸TTS.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

#### Example text to speech using **Fairseq models in ~1100 languages** 🤯.
For Fairseq models, use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
You can find the language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)
and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

```python
# TTS with on the fly voice conversion
api = TTS("tts_models/deu/fairseq/vits")
api.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

### Command-line `tts`

<!-- begin-tts-readme -->

Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

#### Single Speaker Models

- List provided models:

  ```
  $ tts --list_models
  ```

- Get model info (for both tts_models and vocoder_models):

  - Query by type/name:
    The model_info_by_name uses the name as it from the --list_models.
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```
    For example:
    ```
    $ tts --model_info_by_name tts_models/tr/common-voice/glow-tts
    $ tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
    ```
  - Query by type/idx:
    The model_query_idx uses the corresponding idx from --list_models.

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

    For example:

    ```
    $ tts --model_info_by_idx tts_models/3
    ```

  - Query info for model info by full name:
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```
  $ tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --vocoder_name "vocoder_models/en/ljspeech/univnet" --out_path output/path/speech.wav
  ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
      --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker Models

- List the available speakers and choose a <speaker_id> among them:

  ```
  $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```
  $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

### Voice Conversion Models

```
$ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```

<!-- end-tts-readme -->

## Directory Structure
```
|- notebooks/       (Jupyter Notebooks for model evaluation, parameter selection and data analysis.)
|- utils/           (common utilities.)
|- TTS
    |- bin/             (folder for all the executables.)
      |- train*.py                  (train your target model.)
      |- ...
    |- tts/             (text to speech models)
        |- layers/          (model layer definitions)
        |- models/          (model definitions)
        |- utils/           (model specific utilities.)
    |- speaker_encoder/ (Speaker Encoder models.)
        |- (same)
    |- vocoder/         (Vocoder models.)
        |- (same)
```
