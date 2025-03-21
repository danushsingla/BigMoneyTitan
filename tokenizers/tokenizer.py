import regex as re
import unicodedata

# Does not use BPE or any major tokenization algorithm. Rather, simply converts characters to integers and vice versa.
class SimpleTokenizer:
    def __init__(self):
        self.chars = None

    # decoder: take a list of integers, output a string
    def decode(self, ids):
        itos = { i:ch for i,ch in enumerate(self.chars) }
        decoded_text = ''.join(itos[i] for i in ids)
        return decoded_text
    
    # encoders take a string, output a list of integers
    def encode(self, text):
        stoi = { ch:i for i, ch in enumerate(self.chars) }
        encoded_text = [stoi[c] for c in text]
        return encoded_text
    
    def train(self, text):
        print("Training SimpleTokenizer!")
        self.chars = sorted(list(set(text))) # here are all the unique characters that occur in this text
        print("Training finished with vocab size", len(self.chars))
    
    # prints the size of the vocabulary for the embedding layer
    def size(self):
        return len(self.chars)
    
# Base Tokenizer for inheritance
class Tokenizer:
    def __init__(self):
        self.merges = {}        # keeps track of merge list for encode
        # Initialize the vocabulary with the first 256 bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # maps integers to tokens for decode
        self.pattern = "" # pattern if there is one (only for regex and further)
        self.special_tokens = {}  # special tokens if there are some (regex and further or not at all)

    
    def get_stats(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                # continually append tokens unhindered
                newids.append(ids[i])
                i += 1
        return newids
    
    def size(self):
        return len(self.vocab) + len(self.special_tokens)
    
    def replace_control_characters(self, s: str) -> str:
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}") # escape
        return "".join(chars)

    def render_token(self, t: bytes) -> str:
        # pretty print a token, escaping control characters
        s = t.decode('utf-8', errors='replace')
        s = self.replace_control_characters(s)
        return s
    
    def decode(self, ids):
        raise NotImplementedError("Subclasses must implement this method")
    
    def encode(self, text, verbose=False):
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, text):
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        print("Saving tokenizer!")
        # File path
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write version, pattern, and merges
            f.write("tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # write the additional tokens, first the number of them, then each one with index
            f.write(f"{len(self.vocab)-256-len(self.merges)}\n")
            for idx, token in self.vocab.items():
                if idx >= 256 and idx not in self.merges.values() and idx not in self.special_tokens.values():
                    s = self.render_token(token)
                    f.write(f"{s} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges.items():
                f.write(f"{idx1} {idx2}\n")
            # write the vocab for the human to look at
            vocab_file = file_prefix + ".vocab"
            inverted_merges = {idx: pair for pair, idx in self.merges.items()} # merges where idx is the key
            with open(vocab_file, 'w', encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    # decodes each token and replaces control character appropriately
                    s = self.render_token(token)
                    # find the children of this token (essentially unmerge)
                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = self.render_token(self.vocab[idx0])
                        s1 = self.render_token(self.vocab[idx1])
                        # print showcasing the merge operation done from these tokens
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        # otherwise this token was never from a merge so just print it
                        f.write(f"[{s}] {idx}\n")
    
    def load(self, model_file):
        """ Inverse of save() but only for model file """
        print("Loading tokenizer!")
        # make sure we are looking at the right file otherwise this won't work
        assert(model_file.endswith(".model"))
        # read the model file
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "tokenizer v1", f"Unknown version {version}"
            # read pattern
            self.pattern = f.readline().strip()
            # read number of special tokens
            num_special = int(f.readline().strip())
            # read special tokens
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                self.special_tokens[special] = int(special_idx)

                # Make special cases for <pad> and <eos>
                if special == "<eos>":
                    self.eos_token_id = int(special_idx)
                if special == "<pad>":
                    self.pad_token_id = int(special_idx)
            self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}

            # read number of additional tokens
            num_additional = int(f.readline().strip())
            # read additional tokens
            for _ in range(num_additional):
                token, idx = f.readline().strip().split()
                self.vocab[int(idx)] = token.encode("utf-8")
            # read merges
            for line in f:
                # Removes the parantheses and comma from the indices
                idx1, idx2, idx = map(str, line.split())
                idx1 = int(idx1[1:-1])
                idx2 = int(idx2[:-1])
                idx = int(idx)
                self.merges[(idx1, idx2)] = idx
        
        # rebuild the vocab
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]



# Utilizes BPE but at a very basic level
class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))

        # If the number of tokens is just one then it can cause an error as it is looking for a pair
        while len(tokens)>=2:
            stats = self.get_stats(tokens)
            # Want to find the lowest index value in the merges list we need to start with the earliest merges and work our way down
            # the float("inf") is if p cannot be found in merges then it returns infinity which is obviously not going to be a min
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # The above fails if the pair is not in merges so keep track of this condition
            if pair not in self.merges:
                break # nothing else can be merged so break out
            # Finds the correct pair via index then replace in tokens and return
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    # trains the vocabulary for tokenization with some given text and number of merges
    def train(self, text, num_merges=100):
        print("Training BasicTokenizer!")
        # encode the text and mapp it to integers between 0 and 255 (our vocabulary is now 256)
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))

        # Fill vocabulary with merges using helper functions (BPE)
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            # gets the most recurring pair
            pair = max(stats, key=stats.get)
            # map new merges with indices >= 256 since we already have up to 255 index from original characters (no merge list)
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            # preform merge operation with index then add to vocabulary with index
            tokens = self.merge(tokens, pair, idx)
            self.merges[pair] = idx
        
        # Iterates through the merged character dictionary and adds all of the merged indices
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        print("Training finished with vocab size", len(self.vocab))

# Implements splitting pattern for tokenization
class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()

        GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.eos_token_id = None
        self.pad_token_id = None

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:    # special tokens are not in the original vocabulary so this must be checked
                # No need to decode special tokens
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                unk_token = "<unk-" + str(idx) + ">"
                part_bytes.append(unk_token.encode("utf-8"))
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, chunk_bytes):
        """ Encodes a chunk of bytes into a list of token indices """
        # Convert all bytes int o range 0...255
        ids = list(chunk_bytes)

        # If the number of tokens is just one then it can cause an error as it is looking for a pair
        while len(ids)>=2:
            stats = self.get_stats(ids)
            # Want to find the lowest index value in the merges list we need to start with the earliest merges and work our way down
            # the float("inf") is if p cannot be found in merges then it returns infinity which is obviously not going to be a min
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # The above fails if the pair is not in merges so keep track of this condition
            if pair not in self.merges:
                break # nothing else can be merged so break out
            # Finds the correct pair via index then replace in tokens and return
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids
            
    
    def encode_ordinary(self, text):
        """ Ignores special tokens and encodes text normally """
        chunks = re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="all", verbose=False):
        special = None
        if allowed_special == "all":    # if all special tokens are allowed
            special = self.special_tokens
        elif allowed_special == "none":   # no special tokens are allowed, ignore if any are found in text
            special = {}
        elif allowed_special == "none_raise":    # no special tokens are allowed, raise an error if any are found in text
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):      # if allowed_special is a set of specific tokens then allow only those
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}  # creates the mapping
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self.encode_ordinary(text)   # if no special tokens allowed then encode normally
        
        # Separate the special chunks from the other chunks
        special_pattern = "(" + "|".join(re.escape(token) for token in special) + ")"
        special_chunks = re.split(special_pattern, text)

        # all chunks encoded separately then rejoined
        ids = []
        for part in special_chunks:
            if part in special:
                # since this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # encode the ordinary text
                ids.extend(self.encode_ordinary(part))

        if verbose:
            print(f"Encoded {text} into {ids}")
        
        return ids

    
    def register_special_tokens(self, special_tokens):
        # special tokens is dictionary of string -> int
        # example: {"<|endoftext|>": 100257}

        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}

        # Register IDs of each special token
        if "<eos>" in special_tokens:
            self.eos_token_id = special_tokens["<eos>"]
        if "<pad>" in special_tokens:
            self.pad_token_id = special_tokens["<pad>"]
    
    def register_additional_tokens(self, additional_tokens):
        """
        Register extra tokens that I want specifically to be known by the model
        """

        # Add the list addtional_tokens to the end of vocab
        for token in additional_tokens:
            self.vocab[self.size()] = token.encode("utf-8")
    
    def train(self, text, num_merges=100, verbose=False):
        print("Training RegexTokenizer!")
        assert self.size() >= 256, "RegexTokenizer must have at least 256 tokens in the vocabulary"

        # Divides the text into chunks like " thing" or " by"
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # Creates list of bytes per chunk (" thing" is turned into [32, 116, 104, 105, 110, 103])
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Fill vocabulary with merges using helper functions
        for i in range(num_merges):
            # keep track of iterations if it is really long
            if i % 100 == 0:
                print(f"Training merge {i}")
            # keep track of the stats for each token
            stats = {}
            for chunk in ids:
                stats = self.get_stats(chunk, stats)    # updates since each chunk contains a list of bytes
            # gets the most recurring pair
            pair = max(stats, key=stats.get)
            # map new merges with indices >= 256 since we already have up to 255 index from original characters (no merge list)
            idx = self.size() + i
            # perform merge operation with index but only within each chunk
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
        
        # Iterates through the merged character dictionary and adds all of the merged indices
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")
        print("Training finished with vocab size", len(self.vocab))