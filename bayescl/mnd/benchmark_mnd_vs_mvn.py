import torch
from mnd import matrix_normal_kl, matrix_normal_sample
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.utils.benchmark import Timer

torch.set_default_dtype(torch.float64)


def spd_from_seed(size: int, seed: int, jitter: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(size, size, generator=g, dtype=torch.float64)
    return a @ a.mT + jitter * torch.eye(size, dtype=torch.float64)


def bench_case(n: int, p: int) -> dict[str, float | tuple[int, int]]:
    m1 = torch.randn(n, p, dtype=torch.float64)
    m2 = torch.randn(n, p, dtype=torch.float64)
    u1 = spd_from_seed(n, 100 + n)
    u2 = spd_from_seed(n, 200 + n)
    v1 = spd_from_seed(p, 300 + p)
    v2 = spd_from_seed(p, 400 + p)

    globals_ = {
        "torch": torch,
        "MultivariateNormal": MultivariateNormal,
        "kl_divergence": kl_divergence,
        "matrix_normal_sample": matrix_normal_sample,
        "matrix_normal_kl": matrix_normal_kl,
        "m1": m1,
        "m2": m2,
        "u1": u1,
        "u2": u2,
        "v1": v1,
        "v2": v2,
        "n": n,
        "p": p,
    }

    t_sample_mnd = Timer(
        stmt="matrix_normal_sample(m1, u1, v1)",
        globals=globals_,
    ).blocked_autorange(min_run_time=0.4)

    t_sample_mvn = Timer(
        stmt="MultivariateNormal(loc=m1.reshape(-1), covariance_matrix=torch.kron(u1, v1)).sample().reshape(n, p)",
        globals=globals_,
    ).blocked_autorange(min_run_time=0.4)

    t_kl_mnd = Timer(
        stmt="matrix_normal_kl(m1, u1, v1, m2, u2, v2)",
        globals=globals_,
    ).blocked_autorange(min_run_time=0.4)

    t_kl_mvn = Timer(
        stmt="kl_divergence(MultivariateNormal(loc=m1.reshape(-1), covariance_matrix=torch.kron(u1, v1)), MultivariateNormal(loc=m2.reshape(-1), covariance_matrix=torch.kron(u2, v2)))",
        globals=globals_,
    ).blocked_autorange(min_run_time=0.4)

    return {
        "shape": (n, p),
        "sample_mnd_us": t_sample_mnd.median * 1e6,
        "sample_mvn_us": t_sample_mvn.median * 1e6,
        "sample_speedup": t_sample_mvn.median / t_sample_mnd.median,
        "kl_mnd_us": t_kl_mnd.median * 1e6,
        "kl_mvn_us": t_kl_mvn.median * 1e6,
        "kl_speedup": t_kl_mvn.median / t_kl_mnd.median,
    }


def main() -> None:
    print("Performance comparison: mnd vs torch.distributions.MultivariateNormal")
    for n, p in [(2, 2), (4, 3), (8, 8), (12, 10)]:
        result = bench_case(n, p)
        print(f"shape={result['shape']}")
        print(
            f"  sample: mnd={result['sample_mnd_us']:.1f}us  mvn={result['sample_mvn_us']:.1f}us  mvn/mnd={result['sample_speedup']:.2f}x"
        )
        print(
            f"  kl:     mnd={result['kl_mnd_us']:.1f}us      mvn={result['kl_mvn_us']:.1f}us      mvn/mnd={result['kl_speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
