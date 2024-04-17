import torch
from poemai_utils.embeddings.embedder_base import EmbedderBase
from transformers import AutoModel, AutoTokenizer


class SGPTEmbedder(EmbedderBase):
    def __init__(
        self, model_name="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
    ):
        super().__init__()

        self.use_cosine_similarity = True
        self.model_name = model_name
        # Get our models - The package will take care of downloading the models automatically
        # For best performance: Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]

        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

        self.model = AutoModel.from_pretrained(self.model_name)
        # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
        self.model.eval()

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(
            batch_tokens, padding=True, return_tensors="pt"
        )
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(
                **batch_tokens, output_hidden_states=True, return_dict=True
            ).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
            .to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(
            last_hidden_state * input_mask_expanded * weights, dim=1
        )
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def calc_embedding(self, text, is_query: bool = False):
        tokens = self.tokenize_with_specb([text], is_query=is_query)
        return self.get_weightedmean_embedding(tokens)[0]

    def embedding_dim(self):
        return self.model.config.hidden_size
