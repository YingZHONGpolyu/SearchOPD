"""Convert the downloaded Search-R1 parquet files into a minimal verl schema."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert(source, destination, split, limit=None):
    if source.resolve() == destination.resolve():
        raise ValueError("Input data must never be overwritten")
    temporary = destination.with_suffix(".parquet.tmp")
    count = 0
    writer = None
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
            rows = []
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    break
                prompt = row["prompt"]
                if len(prompt) != 1 or prompt[0]["role"] != "user":
                    raise ValueError(f"Expected one plain Search-R1 user prompt at row {count}")
                target = row["reward_model"]["ground_truth"]["target"]
                target = [target] if isinstance(target, str) else list(target)
                rows.append(
                    {
                        "data_source": row["data_source"],
                        "prompt": [{"role": "user", "content": prompt[0]["content"]}],
                        "agent_name": "opd_search_agent",
                        "ability": "fact-retrieval",
                        "reward_model": {"style": "rule", "ground_truth": {"target": target}},
                        "extra_info": {"index": count, "split": split},
                    }
                )
                count += 1
            if rows:
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
            if limit is not None and count >= limit:
                break
        if writer is None:
            raise ValueError("No data rows found")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(destination)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/data/work/YingZhong/data/nq_hotpotqa_train"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/work/YingZhong/data/nq_hotpotqa_opd"))
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit for smoke runs")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        count = convert(args.input_dir / f"{split}.parquet", args.output_dir / f"{split}.parquet", split, args.limit)
        print(f"{split}: {count} rows -> {args.output_dir / (split + '.parquet')}")


if __name__ == "__main__":
    main()
