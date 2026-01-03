from minisbd.modules import Vocab, create_dictionary, TokenizationDataset, output_predictions, get_sentences
from minisbd import SBDetect

detector = SBDetect("onnx/fr.onnx")
print(detector.sentences("I'm not a cat. This is a cat."))
exit(1)

# checkpoint['lexicon']
lexicon = None
dictionary = create_dictionary(lexicon)
