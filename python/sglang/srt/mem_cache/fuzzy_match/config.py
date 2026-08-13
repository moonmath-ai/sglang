# Copyright 2023-2024 SGLang Team
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

"""Configuration for fuzzy prefix matching."""

import msgspec


class FuzzyMatchConfig(msgspec.Struct):
    """Fuzzy matching config. All fuzzy matching is opt-in."""

    # Enable fuzzy prefix matching
    enable_fuzzy_match: bool = False

    # Minimum token span a provider may reuse. Partial exact anchors shorter
    # than this are skipped; zero exact-prefix matches remain eligible.
    fuzzy_min_match_length: int = 16

    # Provider class for fuzzy matching logic. Only 'ExactHash' ships
    # in-tree (content-defined chunking + exact-hash, position-independent
    # reuse for MLA models); the interface admits out-of-tree providers.
    fuzzy_match_provider: str = "ExactHash"

    # Cache fuzzy match results for future reuse
    cache_fuzzy_results: bool = True

    # Skip the provider lookup entirely when the exact-miss suffix is
    # shorter than this. Short suffixes cannot amortize the fuzzy lookup,
    # so this bounds the no-hit overhead on workloads without long
    # reusable content.
    fuzzy_min_suffix_tokens: int = 256

    def __post_init__(self):
        """Validate configuration values."""
        if self.fuzzy_min_match_length < 1:
            raise ValueError(
                f"fuzzy_min_match_length must be >= 1, got {self.fuzzy_min_match_length}"
            )

        if self.fuzzy_match_provider != "ExactHash":
            raise ValueError(
                f"fuzzy_match_provider must be 'ExactHash', "
                f"got {self.fuzzy_match_provider}"
            )

        if self.fuzzy_min_suffix_tokens < 0:
            raise ValueError(
                f"fuzzy_min_suffix_tokens must be >= 0, "
                f"got {self.fuzzy_min_suffix_tokens}"
            )

    @classmethod
    def from_server_args(cls, server_args) -> "FuzzyMatchConfig":
        """Create a config from ServerArgs.

        Only called by the ``fuzzy_match`` radix-cache backend factory, so
        selecting the backend is what enables fuzzy matching; there is no
        separate enable flag.
        """
        return cls(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=server_args.fuzzy_min_match_length,
            fuzzy_match_provider=server_args.fuzzy_match_provider,
        )
