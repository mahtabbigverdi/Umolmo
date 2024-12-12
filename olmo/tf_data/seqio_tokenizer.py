# Copyright 2023 The SeqIO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vocabularies."""

import abc
import dataclasses
import functools
import hashlib
import threading
from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union, List, Tuple

import numpy as np
from absl import logging
import tensorflow.compat.v2 as tf

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

PAD_ID = -1 # -1 for llama tokenizer


class Vocabulary(metaclass=abc.ABCMeta):
    """Abstract class for all vocabularies.

    Subclasses must implement methods for converting between strings and tokens
    both in pure python (`_encode`/`_decode`) and in TensorFlow
    (`_encode_tf`/`_decode_tf`).

    Subclasses are responsible for reserving PAD_ID=0 as well as optionally
    reserving EOS_ID and UNK_ID

    `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
    """

    def __init__(self, extra_ids: int = 0):
        """Vocabulary constructor.

        Args:
          extra_ids: The number of extra IDs to reserve.
        """
        self._extra_ids = extra_ids or 0

    @property
    def bos_token_id(self) -> Optional[int]:
        raise NotImplementedError("need to implement bos_id")

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> Optional[int]:
        raise NotImplementedError("need to implement eos_id")

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    @abc.abstractmethod
    def unk_id(self) -> Optional[int]:
        raise NotImplementedError("need to implement unk_id")

    @property
    def extra_ids(self) -> int:
        return self._extra_ids

    @property
    def vocab_size(self) -> int:
        """Vocabulary size, including extra ids."""
        return self._base_vocab_size + self.extra_ids

    @property
    @abc.abstractmethod
    def _base_vocab_size(self) -> int:
        """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
        # TODO(fjord): add a check that pad_id and unk_id (if present)
        #   are less than _base_vocab_size.
        raise NotImplementedError

    @abc.abstractmethod
    def _encode(self, s: str) -> Sequence[int]:
        raise NotImplementedError

    def encode(self, s: Union[Sequence[int], str]) -> Sequence[int]:
        """Tokenizes string to an int sequence, without adding EOS."""
        return self._encode(s)

    @abc.abstractmethod
    def _decode(self, ids):
        raise NotImplementedError

    def decode(self, ids: Iterable[int], truncate_at_eos=True):
        """Detokenizes int32 iterable to a string, up through first EOS."""
        clean_ids = list(ids)

        if self.unk_id is not None:
            vocab_size = self._base_vocab_size
            clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]

        if truncate_at_eos and (self.eos_token_id is not None and self.eos_token_id in clean_ids):
            clean_ids = clean_ids[: clean_ids.index(self.eos_token_id) + 1]

        return self._decode(clean_ids)

    @abc.abstractmethod
    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
        return self._encode_tf(s)

    @abc.abstractmethod
    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Detokenizes int32 batched Tensor through first EOS."""
        clean_ids = ids

        if self.unk_id is not None:
            base_vocab_size = self._base_vocab_size
            clean_ids = tf.where(
                tf.less(clean_ids, base_vocab_size), clean_ids, self.unk_id
            )

        if self.eos_id is not None:
            # Replace everything after the first eos_id with pad_id.
            after_eos = tf.cumsum(
                tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
                exclusive=True,
                axis=-1,
            )
            clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

        return self._decode_tf(clean_ids)


class PassThroughVocabulary(Vocabulary):
    """Vocabulary that passes through inputs unchanged."""

    def __init__(self, size: int, eos_id: Optional[Any] = None):
        """PassThroughVocabulary constructor.

        Args:
          size: the full size of the vocabulary.
          eos_id: the end-of-sequence token.
        """
        self._size = size
        self._eos_id = eos_id
        super().__init__()

    @property
    def _base_vocab_size(self):
        return self._size

    def _encode(self, s: Sequence[Any]) -> Sequence[Any]:
        return s

    def _decode(self, ids: Sequence[Any]) -> Sequence[Any]:
        return ids

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        return s

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        return ids

    @property
    def eos_id(self) -> Optional[Any]:
        return self._eos_id

    @property
    def unk_id(self) -> Optional[Any]:
        return None

    def __eq__(self, other):
        if not isinstance(other, PassThroughVocabulary):
            return False
        return self._size == other._size and self.eos_id == other.eos_id

    def __str__(self) -> str:
        return f"PassThroughVocabulary(size={self._size}, eos_id={self.eos_id})"


class UnigramVocabulary(Vocabulary):
    """Vocabulary that does table-lookup of unigrams."""

    def __init__(self, unigrams: Sequence[str]):
        """UnigramVocabulary constructor.

        Args:
          unigrams: the collection of in-vocabulary tokens. This collection should
            not include PAD or UNK, which are automatically assigned ids and managed
            as possible decode tokens.
        """

        super().__init__()
        unigrams_as_list = list(unigrams)
        self._unigram_by_id = ["PAD"] + unigrams_as_list + ["UNK"]
        self._id_by_unigram = {u: i for i, u in enumerate(self._unigram_by_id)}
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(["PAD"] + unigrams_as_list),
            # One extra value because the leading 0 corresponds to PAD
            values=tf.constant(range(len(unigrams) + 1), dtype=tf.int64),
        )
        self._id_by_unigram_tf = tf.lookup.StaticVocabularyTable(
            initializer, num_oov_buckets=1
        )
        self._unigram_by_id_tf = tf.constant(self._unigram_by_id)

    def _encode(self, s: str) -> Sequence[int]:
        return [self._id_by_unigram.get(s, self.unk_id)]

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        tf_ids = self._id_by_unigram_tf.lookup(s)
        return tf.expand_dims(tf.dtypes.cast(tf_ids, tf.int32), -1)

    def _decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._unigram_by_id[id] for id in ids)

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        return self._unigram_by_id_tf[ids[0]]

    @property
    def _base_vocab_size(self):
        return len(self._unigram_by_id)

    @property
    def eos_id(self):
        return None

    @property
    def unk_id(self):
        return self._base_vocab_size - 1


class SentencePieceVocabulary(Vocabulary):
    """Wrapper for nlp/sentencepiece encoder.

    Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
    EOS, and ID=2 for UNK.

    If using extra ids, you can represent them in string-form as `<extra_id_0>`,
    `<extra_id_1>`, etc. They will be indexed starting from the end of the
    vocabulary to match how the masking preprocessors are set up.

    IMPORTANT NOTE: these placeholders only work properly when they are used at
    word starts (e.g., "I like peanut butter and <extra_id_0> sandwiches." or
    "I like peanut butter and <extra_id_0>ly sandwiches" are both okay, but
    "I like peanut butter and jel<extra_id_0> sandwiches" is not.).
    """

    @dataclasses.dataclass
    class _ModelContext:
        tokenizer: sentencepiece_processor.SentencePieceProcessor
        sp_model: bytes

    _load_model_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
            self,
            sentencepiece_model_file: str,
            extra_ids: int = 0,
            normalizer_spec_overrides: Optional[
                sentencepiece_model_pb2.NormalizerSpec
            ] = None,
            reverse_extra_ids: bool = False,
            extra_tokens: Tuple[str] = None,
            hack_to_t5_start_tokens: bool = False,
    ):
        """Create a SentencePieceVocabulary.

        Optionally, specify a number of extra ids to add to the end of the
        vocabulary for use as sentinels.

        Args:
          sentencepiece_model_file: path of the sentence piece model.
          extra_ids: number of extra ids to include.
          normalizer_spec_overrides: If not None, this proto will be merged into the
            model's normalizer and denormalizer specs. Thus, any options set on this
            object will override the values of those options in the loaded model.
          reverse_extra_ids: if True, extra_ids are numbered in descending order, so
            the first extra_id has the highest number. This is done for
            compatibility with span_corruption mask generation in T5.
        """
        self._sentencepiece_model_file = sentencepiece_model_file
        self._normalizer_spec_overrides = normalizer_spec_overrides
        self._reverse_extra_ids = reverse_extra_ids
        self._model: Optional[SentencePieceVocabulary._ModelContext] = None
        self._extra_tokens = extra_tokens
        self._hack_to_t5_start_tokens = hack_to_t5_start_tokens
        super().__init__(extra_ids=extra_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Gin config makes a deep copy of the keyword arguments of configurables.
        # When a SentencePieceVocabulary vocabulary is used as a keyword argument
        # in a Gin configurable, it must be picklable. We therefore remove
        # _model; will be initialized lazily as needed.
        del state["_model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

    def load_model(self) -> None:
        _ = self._model_context()

    def _model_context(
            self,
    ) -> _ModelContext:
        """Loads model if not yet loaded and returns the model context.

        Returns:
          The model context as a tuple of (tokenizer, sp_model).
        """
        if self._model:
            return self._model

        normalizer_spec_overrides_serialized = (
            self._normalizer_spec_overrides.SerializeToString(deterministic=True)
            if self._normalizer_spec_overrides
            else None
        )

        self._model = self._load_model(
            self._sentencepiece_model_file,
            self._extra_ids,
            normalizer_spec_overrides_serialized,
            self._reverse_extra_ids,
            extra_tokens=self._extra_tokens,
            hack_to_t5_start_tokens=self._hack_to_t5_start_tokens,
        )
        return self._model

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _load_model(
            cls,
            sentencepiece_model_file: str,
            extra_ids: int,
            normalizer_spec_overrides_serialized: Optional[bytes] = None,
            reverse_extra_ids: bool = True,
            extra_tokens: Tuple[str] = None,
            hack_to_t5_start_tokens=False,
    ) -> _ModelContext:
        """Load SPM, Python tokenizer, and cache results to the class definition."""
        # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
        # Without a lock, users may randomly see SIGSEGV on
        # sentencepiece::ModelInterface::pad_piece when using the vocabulary in
        # SeqIO preprocessors.
        with cls._load_model_lock:
            # Handle cases where SP can't load the file, but gfile can.
            with tf.io.gfile.GFile(sentencepiece_model_file, "rb") as f:
                sp_model = f.read()
                model = sentencepiece_model_pb2.ModelProto.FromString(sp_model)

                if hack_to_t5_start_tokens:
                    # PAD token would still be 0 same as BOS for consistency as previous!
                    unk = model.pieces[0]
                    bos = model.pieces[1]
                    eos = model.pieces[2]
                    model.pieces.remove(unk)
                    model.pieces.remove(bos)
                    model.pieces.remove(eos)
                    model.pieces.insert(0, bos)  # BOS is token 0
                    model.pieces.insert(1, eos)  # EOS is token 1
                    model.pieces.insert(2, unk)  # UNK is token 2

                # Add placeholder strings for extra IDs.
                if extra_ids:
                    # By default, we them in reverse order to match span corruption.
                    if reverse_extra_ids:
                        extra_id_tokens = reversed(range(extra_ids))
                    else:
                        extra_id_tokens = range(extra_ids)

                    for i in extra_id_tokens:
                        model.pieces.add(
                            piece=f"▁<extra_id_{i}>",
                            score=0.0,
                            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
                        )

                if extra_tokens:
                    for s in extra_tokens:
                        model.pieces.add(
                            piece=f"▁"+s,
                            score=0.0,
                            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
                        )
                                    
                if normalizer_spec_overrides_serialized is not None:
                    normalizer_spec_overrides = (
                        sentencepiece_model_pb2.NormalizerSpec.FromString(
                            normalizer_spec_overrides_serialized
                        )
                    )

                    model.normalizer_spec.MergeFrom(normalizer_spec_overrides)
                    model.denormalizer_spec.MergeFrom(normalizer_spec_overrides)
                sp_model = model.SerializeToString()
            # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
            tokenizer = sentencepiece_processor.SentencePieceProcessor()
            tokenizer.LoadFromSerializedProto(sp_model)
            if tokenizer.pad_id() != PAD_ID:
                logging.warning(
                    (
                        "T5 library uses PAD_ID=%s, which is different from the "
                        "sentencepiece vocabulary, which defines pad_id=%s"
                    ),
                    PAD_ID,
                    tokenizer.pad_id(),
                )

            return cls._ModelContext(tokenizer=tokenizer, sp_model=sp_model)

    @property
    def num_extra_tokens(self):
        if self._extra_tokens:
            return len(self._extra_tokens)
        return 0

    @property
    def bos_id(self) -> Optional[int]:
        return self.tokenizer.bos_id()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_id()
    
    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_id()

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_id()
    
    @property
    def unk_id(self) -> Optional[int]:
        return self.tokenizer.unk_id()

    @property
    def sp_model(self) -> Optional[bytes]:
        """Retrieve the SPM."""
        return self._model_context().sp_model

    @property
    def sentencepiece_model_file(self) -> str:
        return self._sentencepiece_model_file

    @property
    def tokenizer(self) -> sentencepiece_processor.SentencePieceProcessor:
        """Returns the Python tokenizer."""
        return self._model_context().tokenizer

    @property
    def tf_tokenizer(self):
        """Instantiate and return a TF tokenizer."""
        import tensorflow_text as tf_text  # import here to keep the dependency optional
        return tf_text.SentencepieceTokenizer(model=self.sp_model)

    @property
    def vocab_size(self):
        return self._base_vocab_size

    @property
    def _base_vocab_size(self):
        """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

        Returns:
          an integer, the vocabulary size
        """
        return self.tokenizer.GetPieceSize()

    def _encode(self, s):
        """Encode a python string as a list of integers.

        Args:
          s: a string

        Returns:
          a list of integers (not terminated by EOS)
        """
        return self.tokenizer.EncodeAsIds(s)

    def _decode(self, ids):
        """Decode a list of integers to a python string.

        Args:
          ids: a list of integers (not terminated by EOS)

        Returns:
          a string
        """
        # convert all the extra ids (sentinels) to UNK=2
        unk_id = self.tokenizer.unk_id()
        piece_size = self.tokenizer.GetPieceSize()
        ids = [unk_id if i >= piece_size else int(i) for i in ids]
        return self.tokenizer.DecodeIds(ids)

    def _encode_tf(self, s):
        """Encode a tf.Scalar string to a tf.Tensor.

        This will be necessary for on-the-fly tokenization.

        Args:
          s: a tf.Scalar with dtype tf.string

        Returns:
          a 1d tf.Tensor with dtype tf.int32
        """
        return self.tf_tokenizer.tokenize(s)

    def _decode_tf(self, ids):
        """Decode in TensorFlow.

        Args:
          ids: a 1d or 2d tf.Tensor with dtype tf.int32

        Returns:
          a 1d or 2d tf.Tensor with dtype tf.string
        """
        return self.tf_tokenizer.detokenize(ids)

    def __eq__(self, other):
        if not isinstance(other, SentencePieceVocabulary):
            return False
        try:
            their_md5 = hashlib.md5(other.sp_model).hexdigest()
        # If other has no sp_model attribute, we can't test for equality
        except AttributeError:
            return False
        if self.sp_model is None:
            return False
        our_md5 = hashlib.md5(self.sp_model).hexdigest()
        return our_md5 == their_md5

    def __str__(self) -> str:
        return (
            f"SentencePieceVocabulary(file={self.sentencepiece_model_file}, "
            f"extra_ids={self._extra_ids}, "
            f"spm_md5={hashlib.md5(self.sp_model).hexdigest()})"
        )

    @property
    def adds_space(self):
        return True


class HfTokenizerWrapper:
    def __init__(self, tokenizer, bos_token_id=None, adds_space=False):
        """
        tokenizer: Tokenizer to wrap
        bos_token_id: BOS token id to use if not `tokenizer.bos_token_id`
        adds_space: If concatenating interdependently tokenized pieces of text, will the tokens
                    already including a seerating space?
        """
        self.adds_space = adds_space
        self.tokenizer = tokenizer
        if bos_token_id is None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            self.bos_token_id = bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_id = -1

    def encode(self, x: str):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def decode(self, x: List[int], truncate_at_eos=True):
        x = [int(t) for t in x]

        if self.eos_token_id == self.bos_token_id and (len(x) > 0 and x[0] == self.eos_token_id):
            # Assume an EOS at the start is functioning as BOS
            x = x[1:]

        if truncate_at_eos:
            # Follow seqio and automatically cut off at EOS
            try:
                eos_ix = x.index(self.eos_token_id)
                x = x[:eos_ix]
            except ValueError:
                pass
        else:
            # Keep our special tokens, but skip BOS/EOS
            x = [t for t in x if t != self.eos_token_id and t != self.bos_token_id]
        return self.tokenizer.decode(x)

    def vocab_size(self):
        return len(self.tokenizer)

    def encode_tf(self, x):
        if isinstance(x, str) or len(x.shape) == 0:
            def _enc(_data):
                _data = _data.item() if isinstance(_data, np.ndarray) else _data
                return self.tokenizer.encode(_data.decode("utf-8"), add_special_tokens=False, return_tensors="np")[0].astype(np.int32)
            return tf.ensure_shape(tf.numpy_function(_enc, [x], tf.int32, stateful=False), [None])

        flattened = tf.reshape(x, [-1])

        def _enc(_data):
            tokens = [self.tokenizer.encode(x.decode("utf-8"), add_special_tokens=False, return_tensors="np")[0].astype(np.int32)
                      for x in _data]
            if len(tokens) == 0:
                return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
            else:
                return np.concatenate(tokens, 0), np.array([len(x) for x in tokens]).astype(np.int32)
        if not (isinstance(x, str) or x.dtype == tf.string):
            raise ValueError("Input be a string or a string numpy array")
        text, lens = tf.numpy_function(_enc, [flattened], (tf.int32, tf.int32), stateful=False)
        lens = tf.ensure_shape(lens, [None])
        text = tf.ensure_shape(text, [None])
        if len(x.shape) == 2:
            n = x.shape[1]
            assert n is not None
            return tf.RaggedTensor.from_nested_row_lengths(
                text,
                [tf.ones(tf.shape(x)[0], dtype=lens.dtype)*n, lens]
            )
        else:
            return tf.RaggedTensor.from_row_lengths(text, lens)


class OLMoTokenizerWrapper(HfTokenizerWrapper):

    def encode(self, x: str):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def encode_tf(self, x):
        if isinstance(x, str) or len(x.shape) == 0:
            def _enc(_data):
                return np.asarray(self.tokenizer.encode(_data.numpy().decode("utf-8"), add_special_tokens=False), dtype=np.int32)
            out = tf.py_function(_enc, (x,), tf.int32)
            return tf.ensure_shape(out, [None])
        else:
            def _enc(_data):
                tokens = [self.tokenizer.encode(x.decode("utf-8"), add_special_tokens=False)
                          for x in _data.numpy()]
                if len(tokens) == 0:
                    return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
                else:
                    return np.concatenate(tokens, 0), np.array([len(x) for x in tokens])
            text, lens = tf.py_function(_enc, (x,), (tf.int32, tf.int32))
            lens = tf.ensure_shape(lens, [None])
            text = tf.ensure_shape(text, [None])
            return tf.RaggedTensor.from_row_lengths(text, lens)
    