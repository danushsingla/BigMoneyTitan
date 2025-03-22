import regex as re
import unicodedata

# Does not use BPE or any major tokenization algorithm. Rather, simply converts characters to integers and vice versa.
class SimpleTokenizer:
    def __init__(self):
        # For Simple Tokenizer, we are going to associate an integer with each unique character from the given text when training
        # Thus the max possible length for this is 26, as there are 26 characters in the alphabet
        self.chars = None

    # Decoder: take a list of integers, output a string
    def decode(self, ids):
        # Get the integer to character mapping
        itos = { i:ch for i,ch in enumerate(self.chars) }
        # Join the characters together to form the decoded text
        decoded_text = ''.join(itos[i] for i in ids)
        return decoded_text
    
    # encoders take a string, output a list of integers
    def encode(self, text):
        # Get the character to integer mapping
        stoi = { ch:i for i, ch in enumerate(self.chars) }
        # Encode the text into a list of integers
        encoded_text = [stoi[c] for c in text]
        return encoded_text
    
    def train(self, text):
        print("Training SimpleTokenizer!")
        self.chars = sorted(list(set(text))) # here are all the unique characters that occur in this text, sorted
        print("Training finished with vocab size", len(self.chars))
    
    # Prints the size of the vocabulary for the embedding layer
    def size(self):
        return len(self.chars)
    
# Base Tokenizer for inheritance, we will merge characters/multiple characters together to create more complex encodings
class Tokenizer:
    def __init__(self):
        self.merges = {}        # keeps track of merge list for encode
        # Initialize the vocabulary with the first 256 bytes (base ASCII characters)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # maps integers to tokens for decode
        self.pattern = "" # pattern if there is one (only for regex and further)
        self.special_tokens = {}  # special tokens if there are some (regex and further or not at all)

    
    def get_stats(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """

        # Initialize a dictionary of counts if it is not already given that will contain the dictionary shown above
        counts = {} if counts is None else counts

        # Iterate through each consecutive pair ( like (1,2) then (2,3) from above) and increment or initialize its place
        # in the dictionary counts if it does not already exist
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """

        # List of new ids with the new token idx replacing all occurrences of pair
        # Should mostly be the same as idx, except for the replacement
        newids = []

        # Iterate through the list of ids and replace each pair with the new index
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
    
    # Return size of the vocabulary, include special tokens as they also have associated indices with them
    def size(self):
        return len(self.vocab) + len(self.special_tokens)
    
    def replace_control_characters(self, s: str) -> str:
        # we don't want to print control characters (like \n, \r, \t, etc.)
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table

        # Final string without control characters
        chars = []

        # Iterate through each character in the string
        for ch in s:
            # If not a control character, append it to chars otherwise ignore it and continue
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}") # escape
        return "".join(chars)
    
    # From a token, return the characters/string that is associated with it
    # Ignore the control characters
    def render_token(self, t: bytes) -> str:
        # pretty print a token, escaping control characters
        s = t.decode('utf-8', errors='replace')
        s = self.replace_control_characters(s)
        return s
    
    # Following three functions are implemented when they are inherited
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
            # Write version, pattern, and merges
            f.write("tokenizer v1\n")
            f.write(f"{self.pattern}\n")

            # Write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            # Write the additional tokens, first the number of them, then each one with index
            # No need to store the base 256 characters, they are universal
            # We also don't include merged tokens nor special tokens
            # The only token we do consider are tokens that are manually added to the vocab, for various purposes depending on
            # the functionality of the tokenizer
            f.write(f"{len(self.vocab)-256-len(self.merges)}\n")
            for idx, token in self.vocab.items():
                if idx >= 256 and idx not in self.merges.values() and idx not in self.special_tokens.values():
                    # Decodes each token and replaces control character appropriately
                    s = self.render_token(token)
                    f.write(f"{s} {idx}\n")

            # The merges dict where, when loading, we can re-merge any tokens necessary
            for idx1, idx2 in self.merges.items():
                f.write(f"{idx1} {idx2}\n")

            # Write the vocab for the human to look at
            vocab_file = file_prefix + ".vocab"

            # Inverted merges, where idx is the key, is easier on the human eye
            inverted_merges = {idx: pair for pair, idx in self.merges.items()}
            with open(vocab_file, 'w', encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    # Decodes each token and replaces control character appropriately
                    s = self.render_token(token)
                    # Find the children of this token (essentially unmerge)
                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = self.render_token(self.vocab[idx0])
                        s1 = self.render_token(self.vocab[idx1])
                        # Print showcasing the merge operation done from these tokens
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        # Otherwise this token was never from a merge so just print it
                        f.write(f"[{s}] {idx}\n")
    
    def load(self, model_file):
        """ Inverse of save() but only for model file """

        print("Loading tokenizer!")

        # Make sure we are looking at the right file otherwise this won't work
        assert(model_file.endswith(".model"))

        # Read the model file
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:

            # Read the version
            version = f.readline().strip()
            assert version == "tokenizer v1", f"Unknown version {version}"

            # Read pattern
            self.pattern = f.readline().strip()

            # Read number of special tokens
            num_special = int(f.readline().strip())

            # Read special tokens
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                self.special_tokens[special] = int(special_idx)

                # Make special cases for <pad> and <eos> (have their own IDs stored as class variables)
                if special == "<eos>":
                    self.eos_token_id = int(special_idx)

                if special == "<pad>":
                    self.pad_token_id = int(special_idx)
            
            # Used for various functionality, mostly to help with regex tokenizer
            self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}

            # Read number of additional tokens
            num_additional = int(f.readline().strip())

            # Read each additional tokens
            for _ in range(num_additional):
                token, idx = f.readline().strip().split()
                self.vocab[int(idx)] = token.encode("utf-8")
            
            # Read merges
            for line in f:
                # Removes the parantheses and comma from the indices
                idx1, idx2, idx = map(str, line.split())
                idx1 = int(idx1[1:-1])
                idx2 = int(idx2[:-1])
                idx = int(idx)

                # Add them to the merges dictionary
                self.merges[(idx1, idx2)] = idx
        
        # Rebuild the vocab based on merges
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]



# Utilizes BPE but at a very basic level
class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def decode(self, ids):
        # Given ids (list of integers), use the vocab class variable to return it as a Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        # Given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))

        # If the number of tokens is just one then it can cause an error as it is looking for a pair
        while len(tokens)>=2:
            # Get the number of times each pair occurs
            stats = self.get_stats(tokens)

            # Want to find the lowest index value in the merges list we need to start with the earliest merges and work our way down
            # The float("inf") is if p cannot be found in merges then it returns infinity which is obviously not going to be a min
            # Ex we have 3: (1,2) and 5: (3,4) then we want to merge 1 and 2 first (since 3 is the lower index value)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # The above fails if the pair is not in merges so keep track of this condition
            if pair not in self.merges:
                break # nothing else can be merged so break out

            # Finds the correct pair via index then replace in tokens and return
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    # Trains the vocabulary for tokenization with some given text and number of merges
    def train(self, text, num_merges=100):
        print("Training BasicTokenizer!")

        # Encode the text and map it to integers between 0 and 255 (our vocabulary is now 256)
        # We are just setting up the first 256 base characters
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))

        # Fill vocabulary with merges using helper functions (BPE)
        for i in range(num_merges):
            # Get occurrences for each consecutive pair
            stats = self.get_stats(tokens)

            # Gets the most recurring pair
            pair = max(stats, key=stats.get)

            # Map new merges with indices >= 256 since we already have up to 255 index from original characters (no merge list)
            idx = 256 + i

            # Perform merge operation with index then add to vocabulary with index
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

        # Actual split patterns for GPT models. We will use GPT4 since it is more descriptive
        GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern

        # Pattern is compiled so it can be used
        self.compiled_pattern = re.compile(self.pattern)

        # Define special tokens, including have IDs special for eos (end of sentence) and pad (padding)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.eos_token_id = None
        self.pad_token_id = None

    def decode(self, ids):
        # All the decoded bytes still in utf-8 format
        part_bytes = []

        # Iterate through each index
        for idx in ids:
            # If the index is already in the vocab (not special token), append its character to part_bytes
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            # Special tokens are not in the original vocabulary so this must be checked
            elif idx in self.inverse_special_tokens:
                # No need to decode special tokens
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            # Otherwise, we found an unknown index (which should never happen if the tokens were generated from this file)
            # This is mostly here for debugging purposes. It will almostnever be used
            # Generate a special token with "unk" as a prefix and the index as a suffix
            else:
                unk_token = "<unk-" + str(idx) + ">"
                part_bytes.append(unk_token.encode("utf-8"))

        # Join the bytes together and decode them into a string
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, chunk_bytes):
        """ Encodes a chunk of bytes into a list of token indices """

        # Convert all bytes int o range 0...255
        ids = list(chunk_bytes)

        # If the number of tokens is just one then it can cause an error as it is looking for a pair
        while len(ids)>=2:
            # Get the number of times each pair occurs
            stats = self.get_stats(ids)

            # Want to find the lowest index value in the merges list we need to start with the earliest merges and work our way down
            # The float("inf") is if p cannot be found in merges then it returns infinity which is obviously not going to be a min
            # Ex we have 3: (1,2) and 5: (3,4) then we want to merge 1 and 2 first (since 3 is the lower index value)
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

        # Get chunks of text based on the pattern
        chunks = re.findall(self.compiled_pattern, text)

        # Initialize list to hold the encoded indices
        ids = []

        # Iterate through each chunk and encode it to utf-8 to convert them into bytes
        # Then call _encode_chunk to convert the bytes into encoded indices
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="all", verbose=False):
        # We can configure which special tokens are used by the allowed_special
        # Default is all since, naturally, we want to use all special tokens
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

        # All chunks encoded separately then rejoined
        ids = []
        for part in special_chunks:
            if part in special:
                # Since this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # Encode the ordinary text
                ids.extend(self.encode_ordinary(part))

        if verbose:
            print(f"Encoded {text} into {ids}")
        
        return ids

    def register_special_tokens(self, special_tokens):
        # Special tokens is dictionary of string -> int
        # Example: {"<|endoftext|>": 100257}

        # Create special tokens and also inverse depending on encoding/decoding
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

        # Fill vocabulary with merges, we keep track of max number of merges we want
        for i in range(num_merges):
            # Keep track of iterations if it is really long
            if i % 100 == 0:
                print(f"Training merge {i}")

            # Keep track of the stats for each chunk
            stats = {}
            for chunk in ids:
                stats = self.get_stats(chunk, stats)    # updates since each chunk contains a list of bytes

            # Gets the most recurring pair
            pair = max(stats, key=stats.get)

            # Map new merges with indices >= 256 since we already have up to 255 index from original characters (no merge list)
            idx = self.size() + i

            # Perform merge operation with index but only within each chunk
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
        
        # Iterates through the merged character dictionary and adds all of the merged indices
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        # Verbose simply means print the stats for each merge
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")
        print("Training finished with vocab size", len(self.vocab))