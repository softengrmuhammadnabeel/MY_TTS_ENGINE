# TTS Models Comprehensive Database

## Text-to-Speech (TTS) Models

### Multilingual Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **xtts_v2** | multilingual | multi-dataset | tts_models | XTTS-v2.0.3 by Coqui with 17 languages support. This is an advanced multilingual text-to-speech model capable of generating high-quality speech in 17 different languages with cross-language voice cloning capabilities. | None | `tts_models/multilingual/multi-dataset/xtts_v2` |
| **xtts_v1.1** | multilingual | multi-dataset | tts_models | XTTS-v1.1 by Coqui with 14 languages, cross-language voice cloning and reference leak fixed. An improved version of the original XTTS model with enhanced voice cloning capabilities and bug fixes for reference leak issues. | None | `tts_models/multilingual/multi-dataset/xtts_v1.1` |
| **your_tts** | multilingual | multi-dataset | tts_models | Your TTS model accompanying the research paper available at https://arxiv.org/abs/2112.02418. This model represents a significant advancement in multilingual TTS technology with zero-shot voice cloning capabilities. | None | `tts_models/multilingual/multi-dataset/your_tts` |
| **bark** | multilingual | multi-dataset | tts_models | 🐶 Bark TTS model released by suno-ai. This innovative model can generate highly realistic speech with various emotions and speaking styles. The original implementation can be found at https://github.com/suno-ai/bark. | None | `tts_models/multilingual/multi-dataset/bark` |

### English Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **tacotron2** | en | ek1 | tts_models | EK1 en-rp tacotron2 by NMStoker. A British English (Received Pronunciation) Tacotron2 model trained on the EK1 dataset, providing high-quality British accent speech synthesis. | `vocoder_models/en/ek1/wavegrad` | `tts_models/en/ek1/tacotron2` |
| **tacotron2-DDC** | en | ljspeech | tts_models | Tacotron2 with Double Decoder Consistency. An enhanced version of Tacotron2 that uses double decoder consistency for improved speech quality and stability during training. | `vocoder_models/en/ljspeech/hifigan_v2` | `tts_models/en/ljspeech/tacotron2-DDC` |
| **tacotron2-DDC_ph** | en | ljspeech | tts_models | Tacotron2 with Double Decoder Consistency with phonemes. This model incorporates phoneme-level processing for more accurate pronunciation and better speech quality. | `vocoder_models/en/ljspeech/univnet` | `tts_models/en/ljspeech/tacotron2-DDC_ph` |
| **glow-tts** | en | ljspeech | tts_models | Glow-TTS model trained on LJSpeech dataset. A flow-based generative model that provides fast and high-quality speech synthesis with improved training stability. | `vocoder_models/en/ljspeech/multiband-melgan` | `tts_models/en/ljspeech/glow-tts` |
| **speedy-speech** | en | ljspeech | tts_models | Speedy Speech model trained on LJSpeech dataset using the Alignment Network for learning the durations. This model focuses on fast inference while maintaining speech quality. | `vocoder_models/en/ljspeech/hifigan_v2` | `tts_models/en/ljspeech/speedy-speech` |
| **tacotron2-DCA** | en | ljspeech | tts_models | Tacotron2 with Decoder Consistency Algorithm. An advanced version of Tacotron2 with improved decoder consistency for better speech synthesis quality. | `vocoder_models/en/ljspeech/multiband-melgan` | `tts_models/en/ljspeech/tacotron2-DCA` |
| **vits** | en | ljspeech | tts_models | VITS is an End2End TTS model trained on LJSpeech dataset with phonemes. A cutting-edge end-to-end TTS model that combines variational inference with adversarial learning for high-quality speech synthesis. | None | `tts_models/en/ljspeech/vits` |
| **vits--neon** | en | ljspeech | tts_models | VITS model with Neon optimizations. An optimized version of VITS for improved performance and efficiency. | None | `tts_models/en/ljspeech/vits--neon` |
| **fast_pitch** | en | ljspeech | tts_models | FastPitch model trained on LJSpeech using the Aligner Network. A non-autoregressive model that provides fast and parallel speech synthesis with controllable pitch and duration. | `vocoder_models/en/ljspeech/hifigan_v2` | `tts_models/en/ljspeech/fast_pitch` |
| **overflow** | en | ljspeech | tts_models | Overflow model trained on LJSpeech dataset. A specialized TTS model designed for handling long-form text synthesis with consistent quality. | `vocoder_models/en/ljspeech/hifigan_v2` | `tts_models/en/ljspeech/overflow` |
| **neural_hmm** | en | ljspeech | tts_models | Neural HMM model trained on LJSpeech dataset. A hybrid model combining neural networks with Hidden Markov Models for robust speech synthesis. | `vocoder_models/en/ljspeech/hifigan_v2` | `tts_models/en/ljspeech/neural_hmm` |
| **vits** | en | vctk | tts_models | VITS End2End TTS model trained on VCTK dataset with 109 different speakers with EN accent. Multi-speaker model capable of generating speech with various English accents and speaker characteristics. | None | `tts_models/en/vctk/vits` |
| **fast_pitch** | en | vctk | tts_models | FastPitch model trained on VCTK dataset. Multi-speaker FastPitch model supporting various English speakers and accents. | None | `tts_models/en/vctk/fast_pitch` |
| **tacotron-DDC** | en | sam | tts_models | Tacotron2 with Double Decoder Consistency trained with Accenture's Sam dataset. Professional-grade TTS model trained on high-quality corporate speech data. | `vocoder_models/en/sam/hifigan_v2` | `tts_models/en/sam/tacotron-DDC` |
| **capacitron-t2-c50** | en | blizzard2013 | tts_models | Capacitron additions to Tacotron 2 with Capacity at 50 as described in https://arxiv.org/pdf/1906.03402.pdf. Enhanced model with improved capacity for handling complex speech patterns. | `vocoder_models/en/blizzard2013/hifigan_v2` | `tts_models/en/blizzard2013/capacitron-t2-c50` |
| **capacitron-t2-c150_v2** | en | blizzard2013 | tts_models | Capacitron additions to Tacotron 2 with Capacity at 150 as described in https://arxiv.org/pdf/1906.03402.pdf. Higher capacity version for even more complex speech synthesis tasks. | `vocoder_models/en/blizzard2013/hifigan_v2` | `tts_models/en/blizzard2013/capacitron-t2-c150_v2` |
| **tortoise-v2** | en | multi-dataset | tts_models | Tortoise TTS model version 2 from https://github.com/neonbjb/tortoise-tts. Advanced TTS model known for extremely high-quality speech synthesis with natural prosody and emotion. | None | `tts_models/en/multi-dataset/tortoise-v2` |
| **jenny** | en | jenny | tts_models | VITS model trained with Jenny(Dioco) dataset. Named as Jenny as demanded by the license. Original model available at https://www.kaggle.com/datasets/noml4u/tts-models--en--jenny-dioco--vits. Single-speaker female voice model. | None | `tts_models/en/jenny/jenny` |

### European Language Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **vits** | bg | cv | tts_models | Bulgarian VITS model trained on Common Voice dataset. High-quality Bulgarian text-to-speech synthesis using the VITS architecture. | None | `tts_models/bg/cv/vits` |
| **vits** | cs | cv | tts_models | Czech VITS model trained on Common Voice dataset. Comprehensive Czech TTS model for natural speech synthesis. | None | `tts_models/cs/cv/vits` |
| **vits** | da | cv | tts_models | Danish VITS model trained on Common Voice dataset. Advanced Danish speech synthesis model. | None | `tts_models/da/cv/vits` |
| **vits** | et | cv | tts_models | Estonian VITS model trained on Common Voice dataset. High-quality Estonian TTS model. | None | `tts_models/et/cv/vits` |
| **vits** | ga | cv | tts_models | Irish Gaelic VITS model trained on Common Voice dataset. Specialized model for Irish language speech synthesis. | None | `tts_models/ga/cv/vits` |
| **tacotron2-DDC** | es | mai | tts_models | Spanish Tacotron2 with Double Decoder Consistency trained on MAI dataset. Professional Spanish TTS model with enhanced consistency. | `vocoder_models/universal/libri-tts/fullband-melgan` | `tts_models/es/mai/tacotron2-DDC` |
| **vits** | es | css10 | tts_models | Spanish VITS model trained on CSS10 dataset. High-quality Spanish speech synthesis model. | None | `tts_models/es/css10/vits` |
| **tacotron2-DDC** | fr | mai | tts_models | French Tacotron2 with Double Decoder Consistency trained on MAI dataset. Professional French TTS model. | `vocoder_models/universal/libri-tts/fullband-melgan` | `tts_models/fr/mai/tacotron2-DDC` |
| **vits** | fr | css10 | tts_models | French VITS model trained on CSS10 dataset. Advanced French speech synthesis model. | None | `tts_models/fr/css10/vits` |
| **glow-tts** | uk | mai | tts_models | Ukrainian Glow-TTS model trained on MAI dataset. Flow-based Ukrainian TTS model. | `vocoder_models/uk/mai/multiband-melgan` | `tts_models/uk/mai/glow-tts` |
| **vits** | uk | mai | tts_models | Ukrainian VITS model trained on MAI dataset. High-quality Ukrainian speech synthesis. | None | `tts_models/uk/mai/vits` |
| **tacotron2-DDC** | nl | mai | tts_models | Dutch Tacotron2 with Double Decoder Consistency trained on MAI dataset. Professional Dutch TTS model. | `vocoder_models/nl/mai/parallel-wavegan` | `tts_models/nl/mai/tacotron2-DDC` |
| **vits** | nl | css10 | tts_models | Dutch VITS model trained on CSS10 dataset. Advanced Dutch speech synthesis model. | None | `tts_models/nl/css10/vits` |
| **tacotron2-DCA** | de | thorsten | tts_models | German Tacotron2 with Decoder Consistency Algorithm trained on Thorsten dataset. High-quality German TTS model. | `vocoder_models/de/thorsten/fullband-melgan` | `tts_models/de/thorsten/tacotron2-DCA` |
| **vits** | de | thorsten | tts_models | German VITS model trained on Thorsten dataset. Advanced German speech synthesis model. | None | `tts_models/de/thorsten/vits` |
| **tacotron2-DDC** | de | thorsten | tts_models | Thorsten-Dec2021-22k-DDC German model with Double Decoder Consistency. Updated German TTS model with improved quality. | `vocoder_models/de/thorsten/hifigan_v1` | `tts_models/de/thorsten/tacotron2-DDC` |
| **vits-neon** | de | css10 | tts_models | German VITS model with Neon optimizations trained on CSS10 dataset. Optimized German TTS model. | None | `tts_models/de/css10/vits-neon` |
| **vits** | hu | css10 | tts_models | Hungarian VITS model trained on CSS10 dataset. High-quality Hungarian speech synthesis. | None | `tts_models/hu/css10/vits` |
| **vits** | el | cv | tts_models | Greek VITS model trained on Common Voice dataset. Advanced Greek TTS model. | None | `tts_models/el/cv/vits` |
| **vits** | fi | css10 | tts_models | Finnish VITS model trained on CSS10 dataset. Comprehensive Finnish speech synthesis model. | None | `tts_models/fi/css10/vits` |
| **vits** | hr | cv | tts_models | Croatian VITS model trained on Common Voice dataset. High-quality Croatian TTS model. | None | `tts_models/hr/cv/vits` |
| **vits** | lt | cv | tts_models | Lithuanian VITS model trained on Common Voice dataset. Advanced Lithuanian speech synthesis. | None | `tts_models/lt/cv/vits` |
| **vits** | lv | cv | tts_models | Latvian VITS model trained on Common Voice dataset. Professional Latvian TTS model. | None | `tts_models/lv/cv/vits` |
| **vits** | mt | cv | tts_models | Maltese VITS model trained on Common Voice dataset. Specialized Maltese speech synthesis model. | None | `tts_models/mt/cv/vits` |
| **vits** | pl | mai_female | tts_models | Polish VITS model with female voice trained on MAI dataset. High-quality Polish female TTS model. | None | `tts_models/pl/mai_female/vits` |
| **vits** | pt | cv | tts_models | Portuguese VITS model trained on Common Voice dataset. Comprehensive Portuguese TTS model. | None | `tts_models/pt/cv/vits` |
| **vits** | ro | cv | tts_models | Romanian VITS model trained on Common Voice dataset. Advanced Romanian speech synthesis. | None | `tts_models/ro/cv/vits` |
| **vits** | sk | cv | tts_models | Slovak VITS model trained on Common Voice dataset. Professional Slovak TTS model. | None | `tts_models/sk/cv/vits` |
| **vits** | sl | cv | tts_models | Slovenian VITS model trained on Common Voice dataset. High-quality Slovenian speech synthesis. | None | `tts_models/sl/cv/vits` |
| **vits** | sv | cv | tts_models | Swedish VITS model trained on Common Voice dataset. Advanced Swedish TTS model. | None | `tts_models/sv/cv/vits` |
| **glow-tts** | it | mai_female | tts_models | Italian Glow-TTS model with female voice as explained in https://github.com/coqui-ai/TTS/issues/1148. Female Italian TTS model with flow-based architecture. | None | `tts_models/it/mai_female/glow-tts` |
| **vits** | it | mai_female | tts_models | Italian VITS model with female voice as explained in https://github.com/coqui-ai/TTS/issues/1148. High-quality female Italian speech synthesis. | None | `tts_models/it/mai_female/vits` |
| **glow-tts** | it | mai_male | tts_models | Italian Glow-TTS model with male voice as explained in https://github.com/coqui-ai/TTS/issues/1148. Male Italian TTS model with flow-based architecture. | None | `tts_models/it/mai_male/glow-tts` |
| **vits** | it | mai_male | tts_models | Italian VITS model with male voice as explained in https://github.com/coqui-ai/TTS/issues/1148. High-quality male Italian speech synthesis. | None | `tts_models/it/mai_male/vits` |
| **glow-tts** | tr | common-voice | tts_models | Turkish GlowTTS model using an unknown speaker from the Common-Voice dataset. High-quality Turkish speech synthesis with flow-based architecture. | `vocoder_models/tr/common-voice/hifigan` | `tts_models/tr/common-voice/glow-tts` |
| **glow-tts** | be | common-voice | tts_models | Belarusian GlowTTS model created by @alex73 (Github). Community-contributed Belarusian TTS model with flow-based architecture. | `vocoder_models/be/common-voice/hifigan` | `tts_models/be/common-voice/glow-tts` |

### Asian Language Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **tacotron2-DDC-GST** | zh-CN | baker | tts_models | Chinese Tacotron2 with Double Decoder Consistency and Global Style Tokens trained on Baker dataset. Advanced Chinese TTS model with style control capabilities. | None | `tts_models/zh-CN/baker/tacotron2-DDC-GST` |
| **tacotron2-DDC** | ja | kokoro | tts_models | Tacotron2 with Double Decoder Consistency trained with Kokoro Speech Dataset. High-quality Japanese TTS model with emotional speech capabilities. | `vocoder_models/ja/kokoro/hifigan_v1` | `tts_models/ja/kokoro/tacotron2-DDC` |

### African Language Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **vits** | ewe | openbible | tts_models | Ewe VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Ewe language. | None | `tts_models/ewe/openbible/vits` |
| **vits** | hau | openbible | tts_models | Hausa VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Hausa language. | None | `tts_models/hau/openbible/vits` |
| **vits** | lin | openbible | tts_models | Lingala VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Lingala language. | None | `tts_models/lin/openbible/vits` |
| **vits** | tw_akuapem | openbible | tts_models | Twi (Akuapem) VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Twi Akuapem dialect. | None | `tts_models/tw_akuapem/openbible/vits` |
| **vits** | tw_asante | openbible | tts_models | Twi (Asante) VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Twi Asante dialect. | None | `tts_models/tw_asante/openbible/vits` |
| **vits** | yor | openbible | tts_models | Yoruba VITS model trained on OpenBible dataset. Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible. Religious text-based TTS model for Yoruba language. | None | `tts_models/yor/openbible/vits` |

### Custom Language Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **vits** | ca | custom | tts_models | Catalan VITS model trained from zero with 101,460 utterances consisting of 257 speakers, approximately 138 hours of speech. Uses three datasets: Festcat, Google Catalan TTS, and Common Voice 8. Trained with TTS v0.8.0. More details at https://github.com/coqui-ai/TTS/discussions/930#discussioncomment-4466345 | None | `tts_models/ca/custom/vits` |
| **glow-tts** | fa | custom | tts_models | Persian TTS female Glow-TTS model for text-to-speech purposes. Single-speaker female voice trained on persian-tts-dataset-female. Note: This model has no compatible vocoder, thus output quality may not be optimal. Dataset available at https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale | None | `tts_models/fa/custom/glow-tts` |
| **vits-male** | bn | custom | tts_models | Single speaker Bangla male VITS model. Comprehensive Bangla TTS model for male voice synthesis. For more information visit https://github.com/mobassir94/comprehensive-bangla-tts | None | `tts_models/bn/custom/vits-male` |
| **vits-female** | bn | custom | tts_models | Single speaker Bangla female VITS model. Comprehensive Bangla TTS model for female voice synthesis. For more information visit https://github.com/mobassir94/comprehensive-bangla-tts | None | `tts_models/bn/custom/vits-female` |

---

## Voice Conversion Models

| Model Name | Language | Dataset | Model Type | Description | Default Vocoder | Model Path |
|------------|----------|---------|------------|-------------|-----------------|------------|
| **freevc24** | multilingual | vctk | voice_conversion_models | FreeVC model trained on VCTK dataset from https://github.com/OlaWod/FreeVC. Advanced voice conversion model capable of converting voice characteristics while preserving linguistic content across multiple languages. | None | `voice_conversion_models/multilingual/vctk/freevc24` |

---

## Vocoder Models

### Universal Vocoders

| Model Name | Language | Dataset | Model Type | Description | Model Path |
|------------|----------|---------|------------|-------------|------------|
| **wavegrad** | universal | libri-tts | vocoder_models | Universal WaveGrad vocoder trained on LibriTTS dataset. High-quality neural vocoder for converting mel-spectrograms to audio waveforms. | `vocoder_models/universal/libri-tts/wavegrad` |
| **fullband-melgan** | universal | libri-tts | vocoder_models | Universal Fullband MelGAN vocoder trained on LibriTTS dataset. Advanced generative vocoder for high-fidelity audio synthesis with full frequency band coverage. | `vocoder_models/universal/libri-tts/fullband-melgan` |

### English Vocoders

| Model Name | Language | Dataset | Model Type | Description | Model Path |
|------------|----------|---------|------------|-------------|------------|
| **wavegrad** | en | ek1 | vocoder_models | EK1 English (Received Pronunciation) WaveGrad vocoder by NMStoker. Specialized vocoder for British English accent synthesis. | `vocoder_models/en/ek1/wavegrad` |
| **multiband-melgan** | en | ljspeech | vocoder_models | Multi-band MelGAN vocoder trained on LJSpeech dataset. Efficient vocoder that processes multiple frequency bands simultaneously for faster inference. | `vocoder_models/en/ljspeech/multiband-melgan` |
| **hifigan_v2** | en | ljspeech | vocoder_models | HiFiGAN v2 LJSpeech vocoder from https://arxiv.org/abs/2010.05646. State-of-the-art generative adversarial network-based vocoder for high-quality audio generation. | `vocoder_models/en/ljspeech/hifigan_v2` |
| **univnet** | en | ljspeech | vocoder_models | UnivNet model fine-tuned on TacotronDDC_ph spectrograms for better compatibility. Universal neural vocoder optimized for phoneme-based TTS models. | `vocoder_models/en/ljspeech/univnet` |
| **hifigan_v2** | en | blizzard2013 | vocoder_models | HiFiGAN v2 vocoder adapted for Blizzard2013 dataset from https://arxiv.org/abs/2010.05646. Professional-grade vocoder for high-quality speech synthesis. | `vocoder_models/en/blizzard2013/hifigan_v2` |
| **hifigan_v2** | en | vctk | vocoder_models | HiFiGAN v2 fine-tuned for VCTK dataset, intended for use with tts_models/en/vctk/sc-glow-tts. Multi-speaker vocoder supporting various English accents. | `vocoder_models/en/vctk/hifigan_v2` |
| **hifigan_v2** | en | sam | vocoder_models | HiFiGAN v2 fine-tuned for SAM dataset, intended for use with tts_models/en/sam/tacotron_DDC. Corporate-grade vocoder for professional speech synthesis. | `vocoder_models/en/sam/hifigan_v2` |

### European Language Vocoders

| Model Name | Language | Dataset | Model Type | Description | Model Path |
|------------|----------|---------|------------|-------------|------------|
| **parallel-wavegan** | nl | mai | vocoder_models | Parallel WaveGAN vocoder for Dutch language trained on MAI dataset. High-quality Dutch speech synthesis vocoder with parallel processing capabilities. | `vocoder_models/nl/mai/parallel-wavegan` |
| **wavegrad** | de | thorsten | vocoder_models | WaveGrad vocoder for German language trained on Thorsten dataset. Diffusion-based vocoder for high-quality German speech synthesis. | `vocoder_models/de/thorsten/wavegrad` |
| **fullband-melgan** | de | thorsten | vocoder_models | Fullband MelGAN vocoder for German language trained on Thorsten dataset. Advanced German vocoder with full frequency band coverage. | `vocoder_models/de/thorsten/fullband-melgan` |
| **hifigan_v1** | de | thorsten | vocoder_models | HiFiGAN v1 vocoder for Thorsten Neutral Dec2021 22k sample rate Tacotron2 DDC model. Specialized German vocoder optimized for the Thorsten dataset. | `vocoder_models/de/thorsten/hifigan_v1` |
| **multiband-melgan** | uk | mai | vocoder_models | Multi-band MelGAN vocoder for Ukrainian language trained on MAI dataset. Ukrainian speech synthesis vocoder with multi-band processing. | `vocoder_models/uk/mai/multiband-melgan` |
| **hifigan** | tr | common-voice | vocoder_models | HiFiGAN vocoder for Turkish language using an unknown speaker from the Common-Voice dataset. High-quality Turkish speech synthesis vocoder. | `vocoder_models/tr/common-voice/hifigan` |
| **hifigan** | be | common-voice | vocoder_models | Belarusian HiFiGAN vocoder created by @alex73 (Github). Community-contributed Belarusian speech synthesis vocoder. | `vocoder_models/be/common-voice/hifigan` |

### Asian Language Vocoders

| Model Name | Language | Dataset | Model Type | Description | Model Path |
|------------|----------|---------|------------|-------------|------------|
| **hifigan_v1** | ja | kokoro | vocoder_models | HiFiGAN v1 vocoder for Japanese language trained on Kokoro dataset by @kaiidams. High-quality Japanese speech synthesis vocoder with emotional speech capabilities. | `vocoder_models/ja/kokoro/hifigan_v1` |

---

## Summary Statistics

- **Total TTS Models**: 72 models
- **Total Voice Conversion Models**: 1 model  
- **Total Vocoder Models**: 18 models
- **Languages Supported**: 40+ languages including multilingual models
- **Architecture Types**: VITS, Tacotron2, Glow-TTS, FastPitch, Bark, XTTS, and more
- **Key Features**: Cross-language voice cloning, multi-speaker support, emotional speech synthesis, and professional-grade quality

## Usage Notes

1. **Model Paths**: Use the exact model paths provided in the tables when loading models
2. **Vocoder Compatibility**: Some models require specific vocoders for optimal performance
3. **Language Support**: Multilingual models support multiple languages in a single model
4. **Quality Levels**: Models vary from research-grade to production-ready quality
5. **Licensing**: Some models have specific licensing requirements (e.g., Jenny model)
6. **Community Contributions**: Several models are contributed by the community (indicated by contributor names)







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
