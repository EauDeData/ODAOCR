from constructors import prepare_model, make_inference, GreedyTextDecoder, CharTokenizer
from PIL import Image


# local_path: dummy_data/
# tokenizer_name: oda_giga_tokenizer
# The tokenizer itself does os.join(local_path/name + .json)
tokenizer = CharTokenizer(False, '/data2/users/amolina/oda_ocr_output/', 'oda_giga_tokenizer')
model = prepare_model(len(tokenizer), device='cpu', load_checkpoint=True, checkpoint_name='/data2/users/amolina/oda_ocr_output/ft_from_hiertext/ftmlt_from_hiertext/ftmlt_from_hiertext.pt')

image = Image.open('/data2/users/amolina/OCR/IIIT5K/test/12_1.png')
print(make_inference(model, tokenizer, GreedyTextDecoder(), image, 'cpu'))