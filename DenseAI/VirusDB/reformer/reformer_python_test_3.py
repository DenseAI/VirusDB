import torch
from transformers import *

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       "bert-base-uncased"),
         ]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights)
#
#     # Encode text
#     input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
#     with torch.no_grad():
#         last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
#
# # Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
# BERT_MODEL_CLASSES = [BertModel]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, force_download=True)