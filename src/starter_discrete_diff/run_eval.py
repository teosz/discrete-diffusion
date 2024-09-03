from pathlib import Path
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel
from lightning.fabric import Fabric
from tqdm import trange
import torch
from datetime import datetime
import pandas as pd
from loguru import logger
import mauve
from vendi_score import vendi
from collections import Counter


"""
Two evals: cond and uncond

For uncond:
    - List all samples
    - Compute AR PPL
    - Save to CSV table

For cond:
    - List all samples
    - Compute mauve, vendi, other diversity metrics

"""


CURR_DATETIME_STR = datetime.now().strftime("%y.%m.%d-%H.%M.%S.%f")
datetime.now().strftime("")


def samples_eval(config):
    uncond_eval(config)
    mauve_eval(config)
    vendi_eval(config)
    diversity_eval(config)


def mauve_eval(config):
    if not config.eval.mauve.run:
        return

    header = [
        "steps",
        "seq_len",
        "prefix_len",
        "add_bos",
        "add_eos",
        "num_samples",
        "from_ema",
        "mauve (mean)",
        "mauve (std)",
    ]

    feature_extractor = AutoModel.from_pretrained("gpt2-large").eval().to("cuda:0")

    cond_parent_path = Path(os.getcwd()) / "samples" / "cond"
    files = list(cond_parent_path.rglob("*.npz"))

    all_result_rows = []
    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)

        metadata = npz_file["metadata"].item()
        references = npz_file["references"]
        samples = npz_file["samples"]

        num_steps = metadata.get("num_steps", "NA")
        seq_len = metadata.get("seq_len", "NA")

        logger.info(f"Computing MAUVE for num_steps={num_steps}, seq_len={seq_len}")
        # Eval on first k tokens
        samples = samples[:, : config.eval.mauve.max_num_tokens]
        references = samples[:, : config.eval.mauve.max_num_tokens]

        q_features = mauve.utils.featurize_tokens_from_model(
            model=feature_extractor,
            tokenized_texts=torch.tensor(samples),
            batch_size=config.eval.mauve.batch_size,
            name="generated samples",
        ).numpy()

        p_features = mauve.utils.featurize_tokens_from_model(
            model=feature_extractor,
            tokenized_texts=torch.tensor(references),
            batch_size=config.eval.mauve.batch_size,
            name="references",
        ).numpy()

        mauve_results = []
        for run_idx in range(config.eval.mauve.num_rounds):
            res = mauve.compute_mauve(
                p_features=p_features,
                q_features=q_features,
                seed=1 + run_idx,
                device_id=0,
                verbose=False,
                batch_size=config.eval.mauve.batch_size,
            ).mauve
            mauve_results.append(float(res))

        mauve_mean = np.mean(mauve_results)
        mauve_std = np.std(mauve_results)

        all_result_rows.append(
            [
                num_steps,
                seq_len,
                metadata.get("prefix_len", "NA"),
                metadata.get("add_bos", "NA"),
                metadata.get("add_eos", "NA"),
                metadata.get("num_samples", "NA"),
                metadata.get("from_ema", "NA"),
                mauve_mean,
                mauve_std,
            ]
        )
    if len(all_result_rows) > 0:
        mauve_res_save_path = (
            Path(os.getcwd()) / f"mauve.csv"
        )
        df = pd.DataFrame(all_result_rows, columns=header)
        df.to_csv(mauve_res_save_path)
        logger.info(f"MAUVE results on conditional samples:\n{df}\n{'=' * 50}")


def _vendi_array(config, ar_model, samples):
    bs = config.eval.vendi.batch_size
    features = mauve.utils.featurize_tokens_from_model(
        model=ar_model,
        tokenized_texts=torch.tensor(samples),
        batch_size=bs,
    )

    if config.eval.vendi.normalize_features:
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)

    sim = features @ features.T
    score = vendi.score_K(sim)
    return score


def vendi_eval(config):
    if not config.eval.vendi.run:
        return

    ar_model = (
        AutoModel.from_pretrained(config.eval.vendi.model)
        .eval()
        .to("cuda:0")
    )

    #######################################
    ##    Eval vendi of uncond samples   ##
    #######################################

    header = [
        "steps",
        "seq_len",
        "add_bos",
        "add_eos",
        "num_samples",
        "from_ema",
        "vendi_samples",
    ]

    uncond_parent_path = Path(os.getcwd()) / "samples" / "uncond"
    files = list(uncond_parent_path.rglob("*.npz"))

    uncond_results = []
    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)
        metadata = npz_file["metadata"].item()
        samples = npz_file["samples"]

        num_steps = metadata.get("num_steps", "NA")
        seq_len = metadata.get("seq_len", "NA")

        logger.info(
            f"Evaluating vendi score for num_steps={num_steps}, seq_len={seq_len}"
        )

        vendi_samples = _vendi_array(config, ar_model, samples)

        curr_res = [
            num_steps,
            seq_len,
            metadata.get("add_bos", "NA"),
            metadata.get("add_eos", "NA"),
            metadata.get("num_samples", "NA"),
            metadata.get("from_ema", "NA"),
            float(vendi_samples),
        ]
        uncond_results.append(curr_res)

    if len(uncond_results) > 0:
        uncond_res_save_path = (
            Path(os.getcwd())
            / "csvs"
            / CURR_DATETIME_STR
            / f"vendi_uncond_w_{config.eval.vendi.model}.csv"
        )
        uncond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(uncond_results, columns=header)
        df.to_csv(uncond_res_save_path)
        logger.info(f"Vendi score of uncond samples:\n{df}\n{'=' * 50}")

    #######################################
    ##    Eval vendi of COND samples   ##
    #######################################

    header = [
        "steps",
        "seq_len",
        "add_bos",
        "add_eos",
        "num_samples",
        "from_ema",
        "vendi_generated",
        "vendi_references",
    ]

    cond_parent_path = Path(os.getcwd()) / "samples" / "cond"
    files = list(cond_parent_path.rglob("*.npz"))

    cond_results = []
    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)
        metadata = npz_file["metadata"].item()
        samples = npz_file["samples"]
        references = npz_file["references"]

        num_steps = metadata.get("num_steps", "NA")
        seq_len = metadata.get("seq_len", "NA")

        logger.info(
            f"Evaluating (cond) vendi score for num_steps={num_steps}, seq_len={seq_len}"
        )

        vendi_generated = _vendi_array(config, ar_model, samples)
        vendi_references = _vendi_array(config, ar_model, references)

        curr_res = [
            num_steps,
            seq_len,
            metadata.get("add_bos"),
            metadata.get("add_eos"),
            metadata.get("num_samples"),
            metadata.get("from_ema"),
            float(vendi_generated),
            float(vendi_references),
        ]
        cond_results.append(curr_res)

    if len(cond_results) > 0:
        cond_res_save_path = (
            Path(os.getcwd())
            / "csvs"
            / CURR_DATETIME_STR
            / f"vendi_cond_w_{config.eval.vendi.model}.csv"
        )
        cond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(cond_results, columns=header)
        df.to_csv(cond_res_save_path)
        logger.info(f"Vendi score of cond samples:\n{df}\n{'=' * 50}")


def compute_entropy(distr):
    out = np.log(distr)
    out = np.where(np.isinf(out), 0.0, out)
    out = -distr * out
    out = np.sum(out, axis=-1)
    return out


def compute_arr_unigram_entropy(arr):
    all_entropies = []

    for row in arr:
        input_ids = row
        indices, counts = np.unique(input_ids, return_counts=True)
        distr = counts / len(input_ids)
        entropy = compute_entropy(distr)
        all_entropies.append(entropy.item())

    m, s = np.mean(all_entropies), np.std(all_entropies)

    return m, s


def compute_repetition_rate(array, rep_l):
    all_repetition_rates = []

    for row in array:
        input_ids = row.tolist()
        mem = Counter(input_ids[:rep_l])
        mem = Counter()

        is_repeated = []

        for rm_idx, new_idx in zip(
            range(-rep_l, len(input_ids)), range(0, len(input_ids))
        ):
            # Check if new token appears in past rep_l tokens
            new_tok = input_ids[new_idx]
            appears = int(mem[new_tok] > 0)
            is_repeated.append(appears)

            if (
                rm_idx >= 0
            ):  # Do not remove while we haven't seen the first rep_l tokens
                to_rm_tok = input_ids[rm_idx]
                mem[to_rm_tok] -= 1

            mem[new_tok] += 1

        all_repetition_rates.append(np.mean(is_repeated))

    m, s = np.mean(all_repetition_rates), np.std(all_repetition_rates)
    return m, s


def compute_unique_unigrams(arr):
    all_unigrams = Counter()
    all_num_unigrams = []

    for row in arr:
        input_ids = row
        all_unigrams.update(input_ids)
        elem_unigrams = Counter(input_ids)
        all_num_unigrams.append(len(elem_unigrams) / len(input_ids))

    m, s = np.mean(all_num_unigrams), np.std(all_num_unigrams)
    return m, s, len(all_unigrams) / sum(all_unigrams.values())


def diversity_eval(config):
    if not config.eval.diversity.run:
        return

    ##################################
    ##    Eval for uncond samples   ##
    ##################################
    header = [
        "steps",
        "seq_len",
        "add_bos",
        "add_eos",
        "num_samples",
        "from_ema",
        "unigr_entropy",
        "rep_rate_20",
        "rep_rate_100",
        "mean_unique_unigr",
    ]

    uncond_results = []
    all_seq_len = set()
    uncond_parent_path = Path(os.getcwd()) / "samples" / "uncond"
    files = list(uncond_parent_path.rglob("*.npz"))

    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)
        metadata = npz_file["metadata"].item()
        samples = npz_file["samples"]

        num_steps = metadata.get("num_steps", "NA")
        seq_len = metadata.get("seq_len", "NA")

        logger.info(
            f"Evaluating diversity (uncond) for num_steps={num_steps}, seq_len={seq_len}"
        )

        unigr_entr, _ = compute_arr_unigram_entropy(samples)
        rep_rate_20, _ = compute_repetition_rate(samples, 20)
        rep_rate_100, _ = compute_repetition_rate(samples, 100)
        mean_unique_unigram, _, _ = compute_unique_unigrams(samples)

        all_seq_len.add(seq_len)
        row = [
            num_steps,
            seq_len,
            metadata.get("add_bos", "NA"),
            metadata.get("add_eos", "NA"),
            metadata.get("num_samples", "NA"),
            metadata.get("from_ema", "NA"),
            float(unigr_entr),
            float(rep_rate_20),
            float(rep_rate_100),
            float(mean_unique_unigram),
        ]
        uncond_results.append(row)

    if len(uncond_results) > 0:
        # Save to file
        uncond_res_save_path = (
            Path(os.getcwd()) / "csvs" / CURR_DATETIME_STR / f"diversity_uncond.csv"
        )
        df = pd.DataFrame(uncond_results, columns=header)
        uncond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(uncond_res_save_path)
        logger.info(f"Diversity results for uncond samples:\n{df}\n{'=' * 50}")

    ##################################
    ##    Eval for uncond samples   ##
    ##################################
    header = [
        "steps",
        "seq_len",
        "add_bos",
        "add_eos",
        "num_samples",
        "from_ema",
        "unigr_entropy (gen)",
        "rep_rate_20 (gen)",
        "rep_rate_100 (gen)",
        "mean_unique_unigr (gen)",
        "unigr_entropy (ref)",
        "rep_rate_20 (ref)",
        "rep_rate_100 (ref)",
        "mean_unique_unigr (ref)",
    ]

    cond_parent_path = Path(os.getcwd()) / "samples" / "cond"
    files = list(cond_parent_path.rglob("*.npz"))
    cond_results = []

    for fname in files:
        npz_file = np.load(fname, allow_pickle=True)
        metadata = npz_file["metadata"].item()
        samples = npz_file["samples"]
        references = npz_file["references"]

        logger.info(
            f"Evaluating diversity (cond) for num_steps={num_steps}, seq_len={seq_len}"
        )

        # Gen
        gen_unigr_entr, _ = compute_arr_unigram_entropy(samples)
        gen_rep_rate_20, _ = compute_repetition_rate(samples, 20)
        gen_rep_rate_100, _ = compute_repetition_rate(samples, 100)
        gen_mean_unique_unigram, _, _ = compute_unique_unigrams(samples)
        # Ref
        ref_unigr_entr, _ = compute_arr_unigram_entropy(references)
        ref_rep_rate_20, _ = compute_repetition_rate(references, 20)
        ref_rep_rate_100, _ = compute_repetition_rate(references, 100)
        ref_mean_unique_unigram, _, _ = compute_unique_unigrams(references)

        # Save results
        curr_res = [
            num_steps,
            seq_len,
            metadata.get("add_bos", "NA"),
            metadata.get("add_eos", "NA"),
            metadata.get("num_samples", "NA"),
            metadata.get("from_ema", "NA"),
            # Gen
            gen_unigr_entr,
            gen_rep_rate_20,
            gen_rep_rate_100,
            gen_mean_unique_unigram,
            # Ref
            ref_unigr_entr,
            ref_rep_rate_20,
            ref_rep_rate_100,
            ref_mean_unique_unigram,
        ]
        cond_results.append(curr_res)

    if len(cond_results) > 0:
        cond_res_save_path = (
            Path(os.getcwd()) / "csvs" / CURR_DATETIME_STR / f"diversity_cond.csv"
        )
        df = pd.DataFrame(cond_results, columns=header)
        cond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cond_res_save_path)
        logger.info(f"Diversity of cond samples:\n{df}\n{'=' * 50}")


def uncond_eval(config):
    uncond_parent_path = Path(os.getcwd()) / "samples" / "uncond"

    files = list(uncond_parent_path.rglob("*.npz"))
    npz_files = [np.load(f, allow_pickle=True) for f in files]
    all_metadata = [f["metadata"].item() for f in npz_files]
    all_metadata_keys = set([k for d in all_metadata for k in d.keys()])
    all_metadata_keys = sorted(list(all_metadata_keys))

    header = all_metadata_keys + ["ar_ppl"]
    model_name = config.eval.ppl_with_ar.model
    ar_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    bs = config.eval.ppl_with_ar.batch_size

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision="32",
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    ar_model = fabric.to_device(ar_model)

    rows = []
    for f in npz_files:
        metadata = f["metadata"].item()
        samples = f["samples"]

        total_loss = 0
        num_examples = 0

        def step_fn(idx):
            batch = samples[idx : idx + bs]
            batch = torch.tensor(batch)
            batch = fabric.to_device(batch)

            with torch.no_grad():
                logits = ar_model(batch).logits[:, :-1]

            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.gather(logits, dim=-1, index=batch[:, 1:, None])[..., 0]

            return loss.mean(-1).sum(), logits.shape[0]

        start = fabric.global_rank * bs
        stop = samples.shape[0]
        step = fabric.world_size * bs
        for idx in trange(
            start, stop, step, desc="Computing AR PPL", disable=fabric.global_rank > 0
        ):
            out = step_fn(idx)
            total_loss += out[0].item()
            num_examples += out[1]

        # Communicate between devices
        total_loss = fabric.all_reduce(torch.tensor([total_loss]), reduce_op="sum")
        num_examples = torch.tensor(num_examples, device=total_loss.device)
        num_examples = fabric.all_reduce(num_examples, reduce_op="sum")

        avg_loss = total_loss / num_examples
        ppl = avg_loss.exp()

        row = [metadata.get(k, "NA") for k in all_metadata_keys] + [float(ppl)]
        rows.append(row)
        fabric.barrier()

    if fabric.global_rank == 0:
        uncond_res_save_path = (
            Path(os.getcwd())
            / "csvs"
            / CURR_DATETIME_STR
            / f"uncond_ppl_w_{config.eval.ppl_with_ar.model}.csv"
        )

        df = pd.DataFrame(rows, columns=header)
        uncond_res_save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(uncond_res_save_path)
        logger.info(f"AR perplexity of uncond samples:\n{df}\n{'=' * 50}")