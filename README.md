# MiniSBD

Free and open source Python library for fast sentence boundary detection. It uses 8bit quantized ONNX models for inference, thus making it fast and lightweight. 

The only dependency is `onnxruntime` / `onnxruntime-gpu`.

## Installation

```bash
pip install -U minisbd
```

## Usage

```python
from minisbd import SBDetect

text = """
La Révolution française (1789-1799) est une période de bouleversements politiques et sociaux en France et dans ses colonies, ainsi qu'en Europe à la fin du XVIIIe siècle. Traditionnellement, on la fait commencer à l'ouverture des États généraux le 5 mai 1789 et finir au coup d'État de Napoléon Bonaparte le 9 novembre 1799 (18 brumaire de l'an VIII). En ce qui concerne l'histoire de France, elle met fin à l'Ancien Régime, notamment à la monarchie absolue, remplacée par la monarchie constitutionnelle (1789-1792) puis par la Première République.
"""

detector = SBDetect("fr", use_gpu=True)
for sent in detector.sentences(text):
    print(f"--> {sent}")

# --> La Révolution française (1789-1799) est une période de bouleversements politiques et sociaux en France et dans ses colonies, ainsi qu'en Europe à la fin du XVIIIe siècle.
# --> Traditionnellement, on la fait commencer à l'ouverture des États généraux le 5 mai 1789 et finir au coup d'État de Napoléon Bonaparte le 9 novembre 1799 (18 brumaire de l'an VIII).
# --> En ce qui concerne l'histoire de France, elle met fin à l'Ancien Régime, notamment à la monarchie absolue, remplacée par la monarchie constitutionnelle (1789-1792) puis par la Première République.
```

By default models are downloaded from GitHub and stored in the user's `~/.cache/minisbd` folder. You can change this at runtime via:

```python
from minisbd import models
models.cache_dir = '/path/to/cache'
```

You can optionally specify a path to a ONNX model instead of having MiniSBD download the model for you:

```python
from minisbd import SBDetect
detector = SBDetect("/path/to/model.onnx")
# ...
```

## Language Support

```python
from minisbd.models import list_models
print(list_models())
```

| Language               | Code    |
| ---------------------- | ------- |
| Afrikaans              | af      |
| Ancient Greek          | grc     |
| Ancient Hebrew         | hbo     |
| Arabic                 | ar      |
| Armenian               | hy      |
| Basque                 | eu      |
| Belarusian             | be      |
| Bulgarian              | bg      |
| Buryat                 | bxr     |
| Catalan                | ca      |
| Chinese (Simplified)   | zh-hans |
| Chinese (Traditional)  | zh-hant |
| Classical Chinese      | lzh     |
| Coptic                 | cop     |
| Croatian               | hr      |
| Czech                  | cs      |
| Danish                 | da      |
| Dutch                  | nl      |
| English                | en      |
| Erzya                  | myv     |
| Estonian               | et      |
| Faroese                | fo      |
| Finnish                | fi      |
| French                 | fr      |
| Galician               | gl      |
| German                 | de      |
| Gothic                 | got     |
| Greek                  | el      |
| Hebrew                 | he      |
| Hindi                  | hi      |
| Hungarian              | hu      |
| Icelandic              | is      |
| Indonesian             | id      |
| Irish                  | ga      |
| Italian                | it      |
| Japanese               | ja      |
| Kazakh                 | kk      |
| Korean                 | ko      |
| Kurmanji               | kmr     |
| Kyrgyz                 | ky      |
| Latin                  | la      |
| Latvian                | lv      |
| Ligurian               | lij     |
| Lithuanian             | lt      |
| Maghrebi Arabic French | qaf     |
| Maltese                | mt      |
| Manx                   | gv      |
| Marathi                | mr      |
| Naija                  | pcm     |
| North Sami             | sme     |
| Norwegian              | nb      |
| Norwegian Nynorsk      | nn      |
| Old Church Slavonic    | cu      |
| Old East Slavic        | orv     |
| Old French             | fro     |
| Persian                | fa      |
| Polish                 | pl      |
| Pomak                  | qpm     |
| Portuguese             | pt      |
| Romanian               | ro      |
| Russian                | ru      |
| Sanskrit               | sa      |
| Scottish Gaelic        | gd      |
| Serbian                | sr      |
| Slovak                 | sk      |
| Slovenian              | sl      |
| Spanish                | es      |
| Swedish                | sv      |
| Tamil                  | ta      |
| Telugu                 | te      |
| Turkish                | tr      |
| Turkish German         | qtd     |
| Ukrainian              | uk      |
| Upper Sorbian          | hsb     |
| Urdu                   | ur      |
| Uyghur                 | ug      |
| Vietnamese             | vi      |
| Welsh                  | cy      |
| Western Armenian       | hyw     |
| Wolof                  | wo      |

## Converting Stanza Models

The `extract.py` script can be used to extract existing Stanza models and convert them to ONNX. See the source code.

## Credits

MiniSBD is a port of [Stanza](https://github.com/stanfordnlp/stanza)'s tokenizer models to ONNX. The models are the same as those from Stanza, but have been converted to ONNX and quantized for faster inference and smaller size.


## License

AGPLv3

Some code has been originally modified from Stanza.