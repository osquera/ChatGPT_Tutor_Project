from transformers import AutoTokenizer, AutoModel
import torch

#for not seing a warning message
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)



def get_text_embedding(text, model_name='bert-base-uncased'):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Get output from pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last layer of output (CLS token) as the text embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()


    return embedding