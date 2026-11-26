"""UTF-8 byte tokenizer with reserved control tokens.

Every Unicode string is represented losslessly as its UTF-8 bytes. This keeps
English, Arabic, emoji, and other scripts supported without a learned vocabulary.
The interface is deliberately small so a subword tokenizer can replace it later.
"""
from __future__ import annotations


class ByteTokenizer:
    PAD = 256
    BOS = 257
    EOS = 258
    VOCAB_SIZE = 259
    SPECIAL_TOKENS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        raw = bytearray()
        for token in ids:
            token = int(token)
            if 0 <= token <= 255:
                raw.append(token)
            elif token in (self.PAD, self.BOS, self.EOS) and skip_special:
                continue
            else:
                raise ValueError(f"token id out of range: {token}")
        # Replacement makes arbitrary generation safe even when the final byte
        # sequence ends in the middle of a multi-byte UTF-8 character.
        return raw.decode("utf-8", errors="replace")
