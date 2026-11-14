# s3-native-front (S6-22s)

A native HTTP front for the GPU processes. It speaks `serving/binproto.py` and writes the hits as the same JSON bytes
as `serving/encode.py`. It is opt-in: the Python front stays the default.

    cargo build --release        # Rust stable; the dependencies' licences are in Cargo.toml
    cargo test --release         # floats against Python's repr (tests/float_vectors.txt)

Run with `S3_SERVING_FRONT_SERVER=native` (`serving/workers.py`). The binary is `target/release/s3-native-front`, or
the path in `S3_NATIVE_FRONT_BIN`.
