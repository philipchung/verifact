"""Patch for Infinity Embeddings v0.0.53

The main modifications made here is to the SentenceTransformerPatched class.

Modifications to init:
- We expect users to load the BAAI/bge-m3 model from Huggingface and we provide
    access to the inner Huggingface AutoModel within SentenceTransformers.
- We also load the weights for the special linear layers required to compute
    sparse and colbert embeddings.

Modifications to encode_core:
- We perform inference forward pass through model to get last hidden states and
    from here we compute dense, sparse, and colbert embeddings. Sparse and colbert
    embeddings are computed using linear layers with weights loaded on initialization.
- Methods are defined for each of the embedding types.
- We delete variables on the GPU to free up memory after processing each batch
    to ensure minimal VRAM usage.

Modifications to encode_post:
- We convert the sparse embeddings to a dictionary of token indices and weights.
- We convert the embeddings to types that are JSON serializable.
- The original method returns a list of embeddings (one embedding item per input sentence).
    Instead, we return a list of dictionaries where each dictionary contains multiple
    embeddings for a single input sentence. The dictionary keys are "dense", "sparse", "colbert"
    and the values are the corresponding embeddings.
- While dense embeddings are a vector, sparse and colbert embeddings are not.
    Sparse embeddings are a key-value mapping, where the key is the token index and
    the value is the token weight. This mapping is created by `process_token_weights` method
    Colbert embeddings are a matrix, where each row corresponds to a token in the input
    sentence and each column corresponds to a feature in the embedding vector.

NOTE: Computing colbert embeddings is slow and compute intensive (20x slower)
    so we have disabled this feature for speed. If you want to enable
    colbert embeddings, uncomment the relevant lines in the SentenceTransformerPatched class.

Original: infinity/libs/infinity_emb/infinity_emb/transformer/embedder/sentence_transformer.py
(https://github.com/michaelfeil/infinity/blob/409f3a28d981354aa77ef7912b418033deb8ed67/libs/infinity_emb/infinity_emb/transformer/embedder/sentence_transformer.py)
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import snapshot_download
from infinity_emb._optional_imports import (  # type: ignore
    CHECK_SENTENCE_TRANSFORMERS,
    CHECK_TORCH,
)
from infinity_emb.args import EngineArgs  # type: ignore
from infinity_emb.log_handler import logger  # type: ignore
from infinity_emb.primitives import Device, Dtype, EmbeddingReturnType  # type: ignore
from infinity_emb.transformer.abstract import BaseEmbedder  # type: ignore
from infinity_emb.transformer.acceleration import to_bettertransformer  # type: ignore
from infinity_emb.transformer.quantization.interface import (  # type: ignore
    quant_embedding_decorator,
    quant_interface,
)

if TYPE_CHECKING:
    from torch import Tensor


if CHECK_SENTENCE_TRANSFORMERS.is_available:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
else:

    class SentenceTransformer:  # type: ignore[no-redef]
        pass


if CHECK_TORCH.is_available:
    import torch
    import torch._dynamo.config
    import torch._inductor.config

    # torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True


class SentenceTransformerPatched(SentenceTransformer, BaseEmbedder):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args=EngineArgs) -> None:
        CHECK_TORCH.mark_required()
        CHECK_SENTENCE_TRANSFORMERS.mark_required()

        model_kwargs = {}
        if engine_args.bettertransformer:
            model_kwargs["attn_implementation"] = "eager"

        ### BGE-M3 Customization Start ###
        # Download model and get reference to local path
        # expect model_name_or_path=BAAI/bge-m3
        model_name_path = snapshot_download(repo_id=engine_args.model_name_or_path)
        ### BGE-M3 Customization End ###

        # Use local downloaded model in cache to initialize class
        super().__init__(
            model_name_or_path=model_name_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=engine_args.device.value,
            model_kwargs=model_kwargs,
        )
        self.to(self.device)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        fm = self._first_module()
        self._infinity_tokenizer = copy.deepcopy(fm.tokenizer)
        self.eval()
        self.engine_args = engine_args

        ### BGE-M3 Customization Start ###
        # To get sparse and colbert embeddings, we need to get access to the
        # underlying inner model within sentence transformers and then load
        # the weights for the linear layers that are used to compute the embeddings.

        # Add Config to Auto Model to Output Last Hidden State
        fm = self._first_module()  # First module is the Transformer model wrapper
        self.model = fm.auto_model  # AutoModel is inner Huggingface Transformers AutoModel
        self.model.config.output_hidden_states = True

        # Setup Linear Layers Sparse Embeddings, Move to GPU
        self.sparse_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size, out_features=1
        )
        sparse_state_dict = torch.load(
            os.path.join(model_name_path, "sparse_linear.pt"), map_location="cpu"
        )
        self.sparse_linear.load_state_dict(sparse_state_dict)
        self.sparse_linear.to(device=self.device)

        # Setup Linear Layers Colbert Embeddings, Move to GPU (Disabled for Speed)
        # colbert_dim: int = -1
        # self.colbert_linear = torch.nn.Linear(
        #     in_features=self.model.config.hidden_size,
        #     out_features=(self.model.config.hidden_size if colbert_dim == -1 else colbert_dim),
        # )
        # colbert_state_dict = torch.load(
        #     os.path.join(model_name_path, 'colbert_linear.pt'), map_location='cpu'
        # )
        # self.colbert_linear.load_state_dict(colbert_state_dict)
        # self.colbert_linear.to(device=self.device)
        ### BGE-M3 Customization End ###

        fm.auto_model = to_bettertransformer(
            fm.auto_model,
            engine_args,
            logger,
        )

        if self.device.type == "cuda" and engine_args.dtype in [
            Dtype.auto,
            Dtype.float16,
        ]:
            logger.info("Switching to half() precision (cuda: fp16). ")
            self.half()
            self.sparse_linear.half()
            # Colbert Embeddings disabled for speed
            # self.colbert_linear.half()

        if engine_args.dtype in (Dtype.int8, Dtype.fp8):
            fm.auto_model = quant_interface(
                fm.auto_model, engine_args.dtype, device=Device[self.device.type]
            )

        if engine_args.compile:
            logger.info("using torch.compile(dynamic=True)")
            fm.auto_model = torch.compile(fm.auto_model, dynamic=True)

    def dense_embedding(self, last_hidden_state: Tensor) -> Tensor:
        return last_hidden_state[:, 0]

    def sparse_embedding(self, last_hidden_state: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.relu(self.sparse_linear(last_hidden_state))

    def colbert_embedding(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * attention_mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def encode_pre(self, sentences) -> dict[str, Tensor]:
        features = self.tokenize(sentences)

        return features

    def encode_core(self, features: Mapping[str, Tensor]) -> dict[Tensor]:
        """
        Computes sentence embeddings.  Single forward pass for a batch.
        Returns a dictionary with "dense_vecs", "sparse_vecs", and "colbert_vecs" keys.
        The corresponding values are the embeddings for each sentence in the batch.

        We apply normalization of dense embeddings by default.
        """
        with torch.no_grad():
            features = util.batch_to_device(features, self.device)
            # Use AutoModel directly instead of SentenceTransformers forward wrapper method
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = model_outputs.last_hidden_state

        ## Compute dense, sparse, colbert vectors.
        # Dense Embeddings
        dense_vecs = self.dense_embedding(last_hidden_state)
        # Normalize dense embeddings
        dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
        # Sparse Embeddings
        sparse_vecs = self.sparse_embedding(last_hidden_state)

        # Colbert Embeddings Disabled for Speed
        # colbert_vecs = self.colbert_embedding(last_hidden_state, attention_mask)
        # colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)

        # Detach, move to CPU, convert to float & numpy array; copy to allow gpu storage to be freed
        out_features = {
            "input_ids": input_ids.detach().cpu().float().numpy().copy(),
            # 'attention_mask': attention_mask.detach().cpu().float().numpy().copy(),
            # 'last_hidden_state': last_hidden_state.detach().cpu().float().numpy().copy(),
            "dense_vecs": dense_vecs.detach().cpu().float().numpy().copy(),
            "sparse_vecs": sparse_vecs.detach().cpu().float().numpy().copy(),
            # 'colbert_vecs': colbert_vecs.detach().cpu().float().numpy().copy(),
        }
        # Delete variables on GPU, free up memory
        del input_ids, attention_mask, model_outputs, last_hidden_state, dense_vecs, sparse_vecs
        torch.cuda.empty_cache()

        return out_features

    # NOTE: quantization currently not supported and will likely fail because
    # quantization _create_statistics_embedding() method expects the output of
    # encode_post to be a list of float vectors, but we are returning a list of
    # dictionaries with multiple embeddings per input sentence.
    @quant_embedding_decorator()
    def encode_post(
        self,
        out_features: dict[str, Tensor | np.ndarray],
        normalize_embeddings: bool = True,
    ) -> list[dict[str, EmbeddingReturnType | Any]]:
        """
        Unpack batch of embeddings in `out_features` (where each key in dict stores
        the whole batch of embeddings) into list of embeddings (where each item in
        the list is a dictionary of embeddings for a single input sentence).

        Convert sparse embeddings to dict of token indices and weights.

        We ignore the `normalize_embeddings` argument because we always normalize
        dense embeddings by default in the upstream `encode_core` method.
        """
        iterables = (
            out_features["input_ids"],
            out_features["dense_vecs"],
            out_features["sparse_vecs"],
            # out_features['colbert_vecs'],
        )

        # Separate batch of embeddings (corresponding to batch of input sentences)
        # so that we have a list with each item corresponds to a single input sentence.
        # Since we have multiple embeddings generated per input sentence, each item
        # in the list is a dictionary of embeddings.
        all_embeddings_list = []
        for input_ids, dense_vec, sparse_vec in zip(*iterables, strict=False):
            # Convert token weights into dictionary of token indices and corresponding weights
            token_weights = sparse_vec.astype(float).squeeze(-1)
            sparse_embeddings = dict(
                self.process_token_weights(
                    token_weights,
                    input_ids.tolist(),
                )
            )
            # Convert embeddings to types that are JSON serializable
            multivector_embedding = {
                "dense": dense_vec.astype(float).tolist(),  # (1024)
                "sparse": sparse_embeddings,  # dict[token_index, weight]
                # 'colbert': colbert_vec.astype(float).tolist(),  # (token seq len, 1024)
            }
            all_embeddings_list.append(multivector_embedding)
        return all_embeddings_list

    def process_token_weights(
        self, token_weights: np.ndarray, input_ids: list
    ) -> defaultdict[Any, int]:
        """Convert sparse token weights into dictionary of token indices and corresponding weights.

        This function is taken from the original FlagEmbedding.bge_m3.BGEM3FlagModel from the
        _process_token_weights() function defined within the encode() method.
        """
        # convert to dict
        result = defaultdict(int)
        unused_tokens = set(
            [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ]
        )
        for w, idx in zip(token_weights, input_ids, strict=False):
            if idx not in unused_tokens and w > 0:
                idx = str(int(idx))
                # w = int(w)
                if w > result[idx]:
                    result[idx] = w
        return result

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]
