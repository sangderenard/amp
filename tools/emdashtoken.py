import argparse

import tiktoken


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List tokens whose decoded form contains a substring."
    )
    parser.add_argument(
        "needle",
        nargs="?",
        default="\u2014",
        help="Substring to search for (default: \\u2014, the em dash).",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    hits = []
    for token in range(enc.n_vocab):
        try:
            # Use the bytes API to avoid errors from special tokens; skip anything undecodable.
            token_bytes = enc.decode_single_token_bytes(token)
        except KeyError:
            continue
        text = token_bytes.decode("utf-8", errors="replace")
        if args.needle in text:
            hits.append((token, text, token_bytes))

    share = len(hits) / enc.n_vocab * 100
    print(f"Tokens containing {args.needle!r}: {len(hits)} ({share:.4f}% of vocab)")
    for tok_id, text, raw in hits:
        print(f"{tok_id:6d} {repr(text)} bytes={raw.hex()}")


if __name__ == "__main__":
    main()
