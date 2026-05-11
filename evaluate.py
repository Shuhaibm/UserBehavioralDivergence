import gc
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import faiss
from scipy.spatial import Voronoi
from loguru import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from umap import UMAP

DISPLAY_NAMES = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4-mini_writing": "GPT-5.4 mini",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "gpt-5.4-nano_writing": "GPT-5.4 nano",
    "claude_4.5-haiku": "Claude Haiku 4.5",
    "claude_4.5_haiku_writing": "Claude Haiku 4.5",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemini-3.1-pro-writing": "Gemini 3.1 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-flash-writing": "Gemini 3 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.1-flash-lite-writing": "Gemini 3.1 Flash-Lite",
    "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "qwen3.5_122b_writing": "Qwen3.5-122B-A10B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "qwen3.5-35b-a3b_writing": "Qwen3.5-35B-A3B",
    "qwen3.5-27b": "Qwen3.5-27B",
    "qwen3.5-27b_writing": "Qwen3.5-27B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.5-9b_writing": "Qwen3.5-9B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-4b_writing": "Qwen3.5-4B",
    "qwen3.5-2b": "Qwen3.5-2B",
    "qwen3.5-2b_writing": "Qwen3.5-2B",
    "qwen3.5-0.8b": "Qwen3.5-0.8B",
    "qwen3.5-0.8b_writing": "Qwen3.5-0.8B",
    "llama3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "llama3.3_70b_writing": "Llama-3.3-70B-Instruct",
    "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "llama3.1_8b_writing": "Llama-3.1-8B-Instruct",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt_oss_120b_writing": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt_oss_20b_writing": "gpt-oss-20b",
    "gemma-4-31b-it": "gemma-4-31B-it",
    "gemma_4_31b_it_writing": "gemma-4-31B-it",
    "gemma-4-26b-a4b-it": "gemma-4-26B-A4B-it",
    "gemma_4_26b_a4b_writing": "gemma-4-26B-A4B-it",
    "gemma-4-e4b-it": "gemma-4-E4B-it",
    "gemma_4_e4b_it_writing": "gemma-4-E4B-it",
    "gemma-4-e2b-it": "gemma-4-E2B-it",
    "gemma_4_e2b_it_writing": "gemma-4-E2B-it",
    "userlm-8b": "UserLM-8b",
    "userlm_8b_writing": "UserLM-8b",
    "humanlm-opinion": "humanlm-opinion",
    "humanlm_opinion_writing": "humanlm-opinion",
}


class Evaluator:
    def __init__(
        self,
        embed_dim: int = 1024,
        k: int = 500,
        max_iter: int = 500,
        n_init: int = 5,
        pca_variance: float = 0.9,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.pca_variance = pca_variance
        self.seed = seed

    def load_truncate_normalize(self, path: str) -> np.ndarray:
        e = np.load(path)["embeddings"][:, :self.embed_dim].astype(np.float64)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(norms, 1e-12)

    def cluster(self, real_emb: np.ndarray, sim_emb: np.ndarray):
        """PCA + FAISS GPU k-means on concatenated embeddings. Returns (combined_pca, real_labels, sim_labels)."""
        combined = np.vstack([real_emb, sim_emb])
        pca = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.seed)
        combined_pca = pca.fit_transform(combined)
        logger.info(f"PCA: {combined.shape[1]}d -> {combined_pca.shape[1]}d ({self.pca_variance*100:.0f}% variance)")

        combined_pca = np.ascontiguousarray(combined_pca, dtype=np.float32)
        d = combined_pca.shape[1]
        kmeans = faiss.Kmeans(d, self.k, nredo=self.n_init, niter=self.max_iter, seed=self.seed, gpu=True)
        kmeans.train(combined_pca)
        _, labels = kmeans.index.search(combined_pca, 1)
        labels = labels.ravel()
        logger.info(f"FAISS K-means (GPU): k={self.k}, niter={self.max_iter}, nredo={self.n_init}")

        n_real = len(real_emb)
        return combined_pca, labels[:n_real], labels[n_real:]

    def compute_divergence_from_labels(self, real_labels: np.ndarray, sim_labels: np.ndarray, n_real: int, n_sim: int) -> dict:
        p_hist = np.bincount(real_labels, minlength=self.k).astype(np.float64) / n_real
        q_hist = np.bincount(sim_labels, minlength=self.k).astype(np.float64) / n_sim

        # Smoothed KL (Laplace smoothing to avoid infinite KL on zero bins)
        alpha = 1.0 / self.k
        p_smooth = (p_hist + alpha)
        p_smooth = p_smooth / p_smooth.sum()
        q_smooth = (q_hist + alpha)
        q_smooth = q_smooth / q_smooth.sum()

        kl_forward = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))
        kl_backward = float(np.sum(q_smooth * np.log(q_smooth / p_smooth)))

        # JS on raw histograms (no smoothing needed — M is always positive where P or Q has mass)
        mask_p = p_hist > 0
        mask_q = q_hist > 0
        m = 0.5 * (p_hist + q_hist)
        kl_p_m = float(np.sum(p_hist[mask_p] * np.log(p_hist[mask_p] / m[mask_p])))
        kl_q_m = float(np.sum(q_hist[mask_q] * np.log(q_hist[mask_q] / m[mask_q])))
        js = 0.5 * kl_p_m + 0.5 * kl_q_m

        metrics = {
            "kl_forward": kl_forward,
            "kl_backward": kl_backward,
            "js_divergence": js,
            "n_real": n_real,
            "n_sim": n_sim,
            "embed_dim": self.embed_dim,
            "k": self.k,
        }

        logger.info(
            f"KL(P||Q)={kl_forward:.4f}  KL(Q||P)={kl_backward:.4f}  JS={js:.4f}  "
            f"(n_real={n_real}, n_sim={n_sim}, dim={self.embed_dim}, k={self.k})"
        )
        return metrics

    def plot_umap_scatter(
        self,
        ax: plt.Axes,
        real_emb: np.ndarray, sim_emb: np.ndarray,
        color_real: str, color_sim: str,
        combined_labels: np.ndarray | None = None,
        title: str = "",
        n_neighbors: int = 100, min_dist: float = 0.0, spread: float = 2.0,
    ):
        combined = np.vstack([real_emb, sim_emb])
        proj = UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
            spread=spread, metric="cosine",
        ).fit_transform(combined)
        real_2d, sim_2d = proj[:len(real_emb)], proj[len(real_emb):]

        ax.scatter(sim_2d[:, 0], sim_2d[:, 1], s=14, alpha=0.4, c=color_sim, linewidths=0, rasterized=True, zorder=2, label="Sim")
        ax.scatter(real_2d[:, 0], real_2d[:, 1], s=14, alpha=0.4, c=color_real, linewidths=0, rasterized=True, zorder=3, label="Real")

        # Voronoi cluster boundaries
        combined_2d = np.vstack([real_2d, sim_2d])
        if combined_labels is not None:
            unique = np.unique(combined_labels)
            centroids = np.array([combined_2d[combined_labels == c].mean(axis=0) for c in unique])
            if len(centroids) >= 4:
                vor = Voronoi(centroids)
                for simplex in vor.ridge_vertices:
                    if -1 not in simplex:
                        pts = vor.vertices[simplex]
                        ax.plot(pts[:, 0], pts[:, 1], color="#888888", linewidth=1.0,
                                alpha=0.6, linestyle="--", zorder=5, clip_on=True)
        pad = 0.05
        xr, yr = np.ptp(combined_2d[:, 0]), np.ptp(combined_2d[:, 1])
        ax.set_xlim(combined_2d[:, 0].min() - pad * xr, combined_2d[:, 0].max() + pad * xr)
        ax.set_ylim(combined_2d[:, 1].min() - pad * yr, combined_2d[:, 1].max() + pad * yr)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        if title:
            ax.set_title(title, fontsize=24)
        ax.legend(markerscale=6, fontsize=20, framealpha=0.8, loc="upper right")

    def plot_pc1_histogram(
        self, ax,
        real_emb: np.ndarray, sim_emb: np.ndarray,
        bins: int, seed: int, ylim: float | None,
        color_real: str, color_sim: str,
    ):
        combined = np.vstack([real_emb, sim_emb])
        pca = PCA(n_components=1, random_state=seed)
        proj = pca.fit_transform(combined).ravel()
        real_proj = proj[:len(real_emb)]
        sim_proj = proj[len(real_emb):]

        ax.hist(real_proj, bins=bins, color=color_real, alpha=0.5, density=True, label="Real")
        ax.hist(sim_proj, bins=bins, color=color_sim, alpha=0.5, density=True, label="Sim")
        ax.set_xlabel("First Principal Component", fontsize=10)
        ax.set_yticks([])
        if ylim:
            ax.set_ylim(0, ylim)
        return real_proj, sim_proj

    def visualize(
        self,
        real_emb: np.ndarray, sim_emb: np.ndarray,
        combined_labels: np.ndarray,
        sim_name: str, output_dir: Path,
        ylim: float | None = None,
        color_real: str = "#0072B2", color_sim: str = "#D55E00",
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        display_name = DISPLAY_NAMES.get(sim_name, sim_name)
        logger.info(f"Computing UMAP and plotting for {display_name}...")
        fig, ax = plt.subplots(figsize=(7, 7))
        self.plot_umap_scatter(ax, real_emb, sim_emb, color_real=color_real, color_sim=color_sim,
                               combined_labels=combined_labels, title=display_name)
        plt.tight_layout()
        plt.savefig(output_dir / "umap.png", dpi=600, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved UMAP scatter: {output_dir / 'umap.png'}")

        fig, ax = plt.subplots(figsize=(7, 3))
        self.plot_pc1_histogram(ax, real_emb, sim_emb, bins=80, seed=self.seed,
                                ylim=ylim, color_real=color_real, color_sim=color_sim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "hist_pc1.png", dpi=600, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved PC1 histogram: {output_dir / 'hist_pc1.png'}")

    def discover_evaluate_visualize(
        self,
        representations_dir: str,
        summary_file: str | None = None,
    ) -> list[dict]:
        rdir = Path(representations_dir)
        pairs = []
        for real_npz in sorted(rdir.glob("**/real_embeddings.npz")):
            annotator_dir = real_npz.parent
            real_source_dir = annotator_dir.parent
            mode_dir = real_source_dir.parent
            annotator_name = annotator_dir.name

            for sim_source_dir in sorted(mode_dir.iterdir()):
                if sim_source_dir == real_source_dir or not sim_source_dir.is_dir():
                    continue
                sim_npz = sim_source_dir / annotator_name / "sim_embeddings.npz"
                if sim_npz.exists():
                    sim_dir = sim_npz.parent
                    mode = mode_dir.name
                    sim_name = sim_source_dir.name
                    label = f"{mode}/{sim_name}"
                    pairs.append((label, str(real_npz), str(sim_npz), sim_name, sim_dir))

        if not pairs:
            logger.warning(f"No embedding pairs found in {representations_dir}")
            return []

        logger.info(f"Found {len(pairs)} embedding pairs")

        # First pass: compute global histogram ylim
        global_ymax = 0
        bins = 80
        for _, real_path, sim_path, _, _ in pairs:
            real_emb = self.load_truncate_normalize(real_path)
            sim_emb = self.load_truncate_normalize(sim_path)
            combined = np.vstack([real_emb, sim_emb])
            proj = PCA(n_components=1, random_state=self.seed).fit_transform(combined).ravel()
            for data in [proj[:len(real_emb)], proj[len(real_emb):]]:
                counts, _ = np.histogram(data, bins=bins, density=True)
                global_ymax = max(global_ymax, counts.max())
        global_ymax *= 1.02

        # Second pass: evaluate and visualize one simulator at a time
        all_results = []
        for label, real_path, sim_path, sim_name, sim_dir in pairs:
            logger.info(f"\n{'='*60}\n{label}\n{'='*60}")
            real_emb = self.load_truncate_normalize(real_path)
            sim_emb = self.load_truncate_normalize(sim_path)

            _, real_labels, sim_labels = self.cluster(real_emb, sim_emb)
            combined_labels = np.concatenate([real_labels, sim_labels])

            metrics = self.compute_divergence_from_labels(real_labels, sim_labels, len(real_emb), len(sim_emb))
            metrics["label"] = label
            all_results.append(metrics)

            output_dir = sim_dir / "divergence_eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "results.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics: {output_dir / 'results.json'}")

            gc.collect()
            if HAS_TORCH:
                torch.cuda.empty_cache()

            self.visualize(real_emb, sim_emb, combined_labels, sim_name, output_dir, ylim=global_ymax)

        if summary_file:
            Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
            with open(summary_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved summary to {summary_file}")

        if len(all_results) > 1:
            logger.info(f"\n{'='*80}")
            logger.info(f"{'Label':<40} {'KL(P||Q)':>10} {'KL(Q||P)':>10} {'JS':>10}")
            logger.info(f"{'='*80}")
            for r in sorted(all_results, key=lambda x: x["js_divergence"]):
                label = r.get("label", "")
                logger.info(f"{label:<40} {r['kl_forward']:>10.4f} {r['kl_backward']:>10.4f} {r['js_divergence']:>10.4f}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize distributional divergence.")
    parser.add_argument("--representations_dir", type=str, required=True, help="Directory with embedding files.")
    parser.add_argument("--summary_file", type=str, default=None, help="Optional path for aggregated results JSON.")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--k", type=int, default=500)
    args = parser.parse_args()

    evaluator = Evaluator(embed_dim=args.embed_dim, k=args.k)
    evaluator.discover_evaluate_visualize(
        args.representations_dir,
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
