# OCR Generalization for Everyone!

## Overview
This repository contains the codebase for the paper **"The OCR Quest for Generalization: Learning to recognize low-resource alphabets with lightweight model editing."** The project focuses on enhancing the generalization ability of OCR models through efficient domain adaptation techniques, enabling recognition of low-resource and unseen alphabets with minimal fine-tuning.

## Paper and Author Information
- **Paper Title:** "The OCR Quest for Generalization: Learning to recognize low-resource alphabets with lightweight model editing."
- **Author:** A. Molina
- **Contact:** [amolina@cvc.uab.cat](mailto:amolina@cvc.uab.cat)
- **Google Scholar:** [https://scholar.google.com/citations?user=23JU52kAAAAJ&hl=en](https://scholar.google.com/citations?user=23JU52kAAAAJ&hl=en)
- **Personal Website:** [https://eaudedata.github.io](https://eaudedata.github.io)
- **Paper Link:** (Provide the link to the paper)

## Features
- +100 trained OCR models available in `./model_card/model_card.html`.
- Model filtering by language, accuracy, and other metrics.
- Pretrained tokenizer and model weights for different scripts.
- Lightweight model editing for OCR adaptation to new alphabets.

## Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running Inference
This repository supports OCR inference using trained models and character tokenizers. Below is an example demonstrating how to use the provided models:

```python
from constructors import prepare_model, make_inference, GreedyTextDecoder, CharTokenizer
from PIL import Image

# Initialize tokenizer
# local_path: dummy_data/
# tokenizer_name: oda_giga_tokenizer
# The tokenizer itself does os.path.join(local_path/name + .json)
tokenizer = CharTokenizer(False, '/data2/users/amolina/oda_ocr_output/', 'oda_giga_tokenizer')

# Load model with tokenizer information
model = prepare_model(len(tokenizer), device='cpu', load_checkpoint=True,
                       checkpoint_name='/data2/users/amolina/oda_ocr_output/ft_from_hiertext/ftmlt_from_hiertext/ftmlt_from_hiertext.pt')

# Load an image and perform OCR
image = Image.open('/data2/users/amolina/OCR/IIIT5K/test/12_1.png')
print(make_inference(model, tokenizer, GreedyTextDecoder(), image, 'cpu'))
```

For additional usage examples, refer to `example.py` in the repository.

## Model Selection
- Access the **model card** at `./model_card/model_card.html`.
- Filter models based on language, accuracy, and architecture.
- Download tokenizer and weights for the selected model.

## License
This project is licensed under **Creative Commons Attribution-ShareAlike (CC BY-SA)**. This means:
- You are free to **share** (copy and redistribute) the material in any medium or format.
- You are free to **adapt** (remix, transform, and build upon) the material.
- **Attribution is required**, and derivatives must be licensed under the same terms.

For further details, refer to [Creative Commons Attribution-ShareAlike](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation
If you use this work, please cite it as follows:

```
@article{Molina2025OCR,
  author = {A. Molina},
  title = {The OCR Quest for Generalization: Learning to recognize low-resource alphabets with lightweight model editing},
  journal = {ICDAR 2025},
  year = {2025},
}
```

## Contact
For questions or collaborations, please reach out to **amolina@cvc.uab.cat**.

