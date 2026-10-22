#!/usr/bin/env python3
"""Execute e analise os 13 blocos do QAOA PEF-24 no IBM Kingston.

O circuito lógico de 24 qubits se fatora exatamente nas componentes conexas do
grafo de interações ZZ. Este programa executa cada componente como um circuito
independente, com os mesmos ângulos do pôster, e reconstrói

    P_sep(ótimo global) = produto_i P_i(padrão ótimo local).

O produto é válido porque o conjunto das 288 coberturas ótimas é exatamente o
produto cartesiano dos conjuntos ótimos locais. O resultado reconstruído não é
uma amostra simultânea de um circuito de 24 qubits; ele é o controle experimental
que isola o efeito de executar/transpilar os blocos separadamente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCE = ROOT / "data" / "frozen_rung1_minimal.json"
DEFAULT_RESULTS = ROOT / "ibm_block_results"
DEFAULT_PLAN = DEFAULT_RESULTS / "kingston_blocks_plan.json"
DEFAULT_ANALYSIS = DEFAULT_RESULTS / "kingston_blocks_analysis.json"
DEFAULT_REPORT = DEFAULT_RESULTS / "kingston_blocks_report.md"
VALIDATION_PLAN = DEFAULT_RESULTS / "ideal_validation_plan.json"
VALIDATION_ANALYSIS = DEFAULT_RESULTS / "ideal_validation_analysis.json"
VALIDATION_REPORT = DEFAULT_RESULTS / "ideal_validation_report.md"

BACKEND = "ibm_kingston"
GAMMA = -0.04
BETA = 0.35
DEFAULT_SHOTS = 10_000
REFERENCE = {
    "full_job_id": "dakjch39k43c73aik9rg",
    "full_shots": 100_000,
    "full_optimal_hits": 79,
    "full_p_opt": 79 / 100_000,
    "ideal_p_opt": 0.001825377751751246,
    "ideal_expected_hits_at_100k": 182.5377751751246,
    "full_physical_two_qubit_gates": 49,
    "full_physical_depth": 64,
    "full_measured_physical_qubits": 24,
    "full_independent_error_proxy": 0.8166713214196704,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def runtime_service():
    """Carrega a conta IBM e contorna `instance=auto` em clientes Runtime antigos."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        return QiskitRuntimeService()
    except Exception:
        accounts = QiskitRuntimeService.saved_accounts()
        auto_accounts = [account for account in accounts.values() if account.get("instance") == "auto"]
        if len(auto_accounts) != 1:
            raise
        account = auto_accounts[0]
        return QiskitRuntimeService(
            channel=account.get("channel", "ibm_quantum_platform"),
            token=account["token"],
            instance=None,
        )


def load_instance(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    data["A"] = np.asarray(data["A"], dtype=np.int16)
    data["h"] = np.asarray(data["h"], dtype=float)
    data["J"] = {
        tuple(int(x) for x in key.split(",")): float(value)
        for key, value in data["J"].items()
    }
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    if data["A"].shape != (data["n_t"], data["n_b"]):
        raise ValueError("dimensões de A incompatíveis com n_t e n_b")
    return data


def connected_blocks(n: int, interactions: dict[tuple[int, int], float]) -> list[list[int]]:
    adjacency = [set() for _ in range(n)]
    for i, j in interactions:
        adjacency[i].add(j)
        adjacency[j].add(i)
    blocks: list[list[int]] = []
    unseen = set(range(n))
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            q = stack.pop()
            component.append(q)
            for other in sorted(adjacency[q], reverse=True):
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        blocks.append(sorted(component))
    return blocks


def optimal_local_patterns(
    instance: dict[str, Any], blocks: list[list[int]], chunk: int = 1 << 18
) -> tuple[list[set[tuple[int, ...]]], int]:
    """Projeta todas as coberturas globais ótimas nos blocos, em memória limitada."""
    A = instance["A"]
    n = int(instance["n_b"])
    k_opt = int(instance["k_opt"])
    target = np.minimum(int(instance["m"]), A.sum(axis=1))
    patterns: list[set[tuple[int, ...]]] = [set() for _ in blocks]
    n_optima = 0
    shifts = np.arange(n, dtype=np.uint32)
    for start in range(0, 1 << n, chunk):
        indices = np.arange(start, min(start + chunk, 1 << n), dtype=np.uint32)
        bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        selected = bits.sum(axis=1) == k_opt
        if not np.any(selected):
            continue
        candidates = bits[selected]
        feasible = np.all(candidates.astype(np.int16) @ A.T >= target[None, :], axis=1)
        optima = candidates[feasible]
        n_optima += len(optima)
        for local, qubits in zip(patterns, blocks):
            local.update(tuple(row) for row in optima[:, qubits].tolist())
    product_size = math.prod(len(local) for local in patterns)
    if product_size != n_optima:
        raise RuntimeError(
            f"as soluções ótimas não fatoram: produto={product_size}, total={n_optima}"
        )
    return patterns, n_optima


def feasible_local_patterns(
    instance: dict[str, Any], blocks: list[list[int]]
) -> list[set[tuple[int, ...]]]:
    """Enumera os padrões que satisfazem as restrições pertencentes a cada bloco."""
    A = instance["A"]
    target = np.minimum(int(instance["m"]), A.sum(axis=1))
    owner: dict[int, int] = {}
    for row in range(A.shape[0]):
        support = set(np.flatnonzero(A[row]).tolist())
        owners = [i for i, block in enumerate(blocks) if support.intersection(block)]
        if len(owners) != 1 or not support.issubset(set(blocks[owners[0]])):
            raise RuntimeError(f"a restrição {row} atravessa componentes ZZ")
        owner[row] = owners[0]
    output: list[set[tuple[int, ...]]] = []
    for block_index, block in enumerate(blocks):
        rows = [row for row, index in owner.items() if index == block_index]
        local_A = A[np.ix_(rows, block)]
        local_target = target[rows]
        accepted: set[tuple[int, ...]] = set()
        for index in range(1 << len(block)):
            x = np.array([(index >> q) & 1 for q in range(len(block))], dtype=np.int8)
            if np.all(local_A @ x >= local_target):
                accepted.add(tuple(int(v) for v in x))
        output.append(accepted)
    return output


def qaoa_block_circuit(
    instance: dict[str, Any], block: list[int], gamma: float, beta: float, measure: bool
):
    from qiskit import QuantumCircuit

    local_index = {global_q: local_q for local_q, global_q in enumerate(block)}
    circuit = QuantumCircuit(len(block), name="block_" + "_".join(map(str, block)))
    circuit.h(range(len(block)))
    for local_q, global_q in enumerate(block):
        hq = float(instance["h"][global_q])
        if hq != 0.0:
            circuit.rz(2.0 * gamma * hq, local_q)
    for (i, j), coupling in instance["J"].items():
        if i in local_index and j in local_index:
            circuit.rzz(2.0 * gamma * coupling, local_index[i], local_index[j])
    circuit.rx(2.0 * beta, range(len(block)))
    if measure:
        circuit.measure_all()
    return circuit


def state_probabilities(instance: dict[str, Any], block: list[int], gamma: float, beta: float):
    from qiskit.quantum_info import Statevector

    circuit = qaoa_block_circuit(instance, block, gamma, beta, measure=False)
    return np.asarray(Statevector(circuit).probabilities(), dtype=float)


def pattern_index(pattern: Iterable[int]) -> int:
    """Qiskit: o qubit local 0 é o bit menos significativo do índice."""
    return sum(int(bit) << q for q, bit in enumerate(pattern))


def pattern_text(pattern: Iterable[int]) -> str:
    """Ordem lógica x[q_global] crescente, não a ordem visual invertida do Qiskit."""
    return "".join(str(int(bit)) for bit in pattern)


def ideal_block_rows(instance: dict[str, Any], gamma: float, beta: float):
    from qiskit import transpile

    blocks = connected_blocks(int(instance["n_b"]), instance["J"])
    optimal, n_optima = optimal_local_patterns(instance, blocks)
    feasible = feasible_local_patterns(instance, blocks)
    rows: list[dict[str, Any]] = []
    circuits = []
    for index, (block, opt_patterns, feas_patterns) in enumerate(
        zip(blocks, optimal, feasible), start=1
    ):
        probabilities = state_probabilities(instance, block, gamma, beta)
        p_opt = float(sum(probabilities[pattern_index(p)] for p in opt_patterns))
        p_feasible = float(sum(probabilities[pattern_index(p)] for p in feas_patterns))
        measured = qaoa_block_circuit(instance, block, gamma, beta, measure=True)
        decomposed = transpile(
            measured,
            basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=1,
            seed_transpiler=73421,
        )
        block_set = set(block)
        n_rzz = sum(1 for i, j in instance["J"] if i in block_set and j in block_set)
        row = {
            "tag": f"block_{index:02d}",
            "original_qubits": block,
            "n_qubits": len(block),
            "optimal_patterns_logical_order": sorted(pattern_text(p) for p in opt_patterns),
            "feasible_patterns_logical_order": sorted(pattern_text(p) for p in feas_patterns),
            "n_optimal_patterns": len(opt_patterns),
            "n_feasible_patterns": len(feas_patterns),
            "p_opt_ideal": p_opt,
            "p_feasible_ideal": p_feasible,
            "uniform_p_opt": len(opt_patterns) / (1 << len(block)),
            "logical_rzz_gates": n_rzz,
            "logical_cx_equivalent": 2 * n_rzz,
            "basis_decomposed_two_qubit_gates": sum(
                1 for instruction in decomposed.data if instruction.operation.num_qubits == 2
            ),
            "logical_depth": decomposed.depth(),
            "ideal_probabilities_qiskit_index": probabilities.tolist(),
        }
        rows.append(row)
        circuits.append(measured)
    product = float(math.prod(row["p_opt_ideal"] for row in rows))
    feasible_product = float(math.prod(row["p_feasible_ideal"] for row in rows))
    if not math.isclose(product, REFERENCE["ideal_p_opt"], rel_tol=0.0, abs_tol=2e-14):
        raise RuntimeError(
            f"validação ideal falhou: produto={product:.16g}, referência={REFERENCE['ideal_p_opt']:.16g}"
        )
    return circuits, rows, n_optima, product, feasible_product


def make_plan(instance_path: Path, backend: str, shots: int) -> tuple[dict[str, Any], list[Any]]:
    instance = load_instance(instance_path)
    angles = instance.get("qaoa_angles", {}).get("1", {})
    gamma = float(angles.get("gamma", [GAMMA])[0])
    beta = float(angles.get("beta", [BETA])[0])
    circuits, rows, n_optima, p_ideal, p_feasible = ideal_block_rows(instance, gamma, beta)
    plan = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "prepared",
        "experiment": "PEF-24 QAOA p=1 executado como 13 componentes ZZ independentes",
        "interpretation": (
            "controle por blocos; o produto reconstruído não é uma execução simultânea "
            "do circuito físico de 24 qubits"
        ),
        "backend": backend,
        "shots_per_block": shots,
        "n_blocks": len(rows),
        "total_circuit_shots": shots * len(rows),
        "runtime_options": {
            "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
            "twirling": {"enable_gates": True, "enable_measure": True},
            "transpile_optimization_level": 3,
            "seed_transpiler": 73421,
        },
        "instance": {
            "path": str(instance_path.resolve()),
            "sha256": instance["sha256"],
            "n_b": int(instance["n_b"]),
            "n_t": int(instance["n_t"]),
            "m": int(instance["m"]),
            "k_opt": int(instance["k_opt"]),
            "n_ZZ": len(instance["J"]),
        },
        "angles": {"p": 1, "gamma": gamma, "beta": beta},
        "factorisation_check": {
            "n_optimal_global_solutions": n_optima,
            "product_of_local_optimum_counts": math.prod(
                row["n_optimal_patterns"] for row in rows
            ),
            "optima_factorise": True,
            "ideal_product_p_opt": p_ideal,
            "ideal_product_p_feasible": p_feasible,
        },
        "reference": REFERENCE,
        "blocks": rows,
    }
    return plan, circuits


def print_plan(plan: dict[str, Any]) -> None:
    print(
        f"{'bloco':>8} {'qubits originais':>22} {'n':>2} {'CXeq':>4} "
        f"{'Popt ideal':>12} {'ótimos locais':>15}"
    )
    for row in plan["blocks"]:
        print(
            f"{row['tag']:>8} {str(row['original_qubits']):>22} {row['n_qubits']:>2} "
            f"{row['logical_cx_equivalent']:>4} {row['p_opt_ideal']:>12.8f} "
            f"{','.join(row['optimal_patterns_logical_order']):>15}"
        )
    fact = plan["factorisation_check"]
    print(f"\nblocos: {plan['n_blocks']}")
    print(f"shots por bloco: {plan['shots_per_block']:,}")
    print(f"shots de circuitos no total: {plan['total_circuit_shots']:,}")
    print(f"soluções ótimas globais: {fact['n_optimal_global_solutions']}")
    print(f"P_ideal(ótimo), produto dos blocos: {fact['ideal_product_p_opt']:.12g}")
    print(f"hits ideais equivalentes em 100.000: {100_000 * fact['ideal_product_p_opt']:.3f}")


def measured_physical_qubits(circuit: Any) -> list[int]:
    return sorted(
        {
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "measure"
        }
    )


def command_prepare(args: argparse.Namespace) -> None:
    plan, _ = make_plan(args.instance, args.backend, args.shots)
    write_json(args.plan, plan)
    print_plan(plan)
    print(f"\nPlano validado e salvo em {args.plan}")


def command_submit(args: argparse.Namespace) -> None:
    from qiskit import transpile
    from qiskit_ibm_runtime import Batch, SamplerV2

    if args.plan.exists() and not args.force_new:
        old = load_json(args.plan)
        if old.get("job_id"):
            raise SystemExit(
                f"o plano já contém o job {old['job_id']}; use --force-new apenas para repetir"
            )
    plan, circuits = make_plan(args.instance, args.backend, args.shots)
    print_plan(plan)
    write_json(args.plan, plan)
    print("\nAutenticando e consultando o backend...")
    try:
        service = runtime_service()
        backend = service.backend(args.backend)
    except Exception as exc:
        plan["status"] = "account_or_backend_check_failed"
        plan["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.plan, plan)
        raise SystemExit(
            f"não foi possível autenticar/consultar {args.backend}: "
            f"{type(exc).__name__}: {exc}"
        ) from None
    backend_status = backend.status()
    if not backend_status.operational:
        raise RuntimeError(f"backend não operacional: {backend_status.status_msg}")
    print(
        f"{backend.name}: operacional; fila informada={backend_status.pending_jobs}; "
        f"status={backend_status.status_msg}"
    )

    transpiled = transpile(
        circuits,
        backend=backend,
        optimization_level=3,
        seed_transpiler=plan["runtime_options"]["seed_transpiler"],
    )
    for row, circuit in zip(plan["blocks"], transpiled):
        row["physical_depth"] = circuit.depth()
        row["physical_two_qubit_gates"] = sum(
            1 for instruction in circuit.data if instruction.operation.num_qubits == 2
        )
        row["physical_operations"] = dict(circuit.count_ops())
        row["measured_physical_qubits"] = measured_physical_qubits(circuit)
    plan["backend_status_at_submission"] = {
        "operational": bool(backend_status.operational),
        "pending_jobs": int(backend_status.pending_jobs),
        "status_msg": str(backend_status.status_msg),
    }
    plan["status"] = "transpiled_not_submitted"
    write_json(args.plan, plan)

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        job = sampler.run(transpiled, shots=args.shots)
    plan["job_id"] = job.job_id()
    plan["submitted_utc"] = utc_now()
    plan["status"] = "submitted"
    write_json(args.plan, plan)
    print(f"\nJob submetido: {plan['job_id']}")
    print(f"Plano atualizado: {args.plan}")
    print(f"Depois de concluído: {sys.executable} {Path(__file__).name} analyze")


def command_status(args: argparse.Namespace) -> None:
    plan = load_json(args.plan)
    job_id = args.job_id or plan.get("job_id")
    if not job_id:
        raise SystemExit("nenhum job_id no plano; informe --job-id")
    job = runtime_service().job(job_id)
    print(f"job_id: {job_id}")
    print(f"status: {job.status()}")
    print(f"backend: {job.backend().name}")
    print(f"criado: {job.creation_date}")


def clean_count_key(bitstring: str) -> str:
    return bitstring.replace(" ", "").replace("_", "")


def logical_pattern_from_qiskit(bitstring: str, n: int) -> tuple[int, ...]:
    clean = clean_count_key(bitstring)
    if len(clean) != n or set(clean) - {"0", "1"}:
        raise ValueError(f"bitstring inesperada para {n} qubits: {bitstring!r}")
    return tuple(int(bit) for bit in clean[::-1])


def counts_to_probabilities(counts: dict[str, int], n: int) -> np.ndarray:
    shots = sum(int(value) for value in counts.values())
    probabilities = np.zeros(1 << n, dtype=float)
    for bitstring, value in counts.items():
        pattern = logical_pattern_from_qiskit(bitstring, n)
        probabilities[pattern_index(pattern)] += int(value) / shots
    return probabilities


def analyze_count_list(
    plan: dict[str, Any], count_list: list[dict[str, int]], posterior_draws: int = 200_000
) -> dict[str, Any]:
    if len(count_list) != len(plan["blocks"]):
        raise ValueError(
            f"foram recebidos {len(count_list)} circuitos; o plano exige {len(plan['blocks'])}"
        )
    block_results: list[dict[str, Any]] = []
    for metadata, raw_counts in zip(plan["blocks"], count_list):
        counts = {clean_count_key(k): int(v) for k, v in raw_counts.items()}
        shots = sum(counts.values())
        optimal = {
            tuple(int(bit) for bit in text)
            for text in metadata["optimal_patterns_logical_order"]
        }
        feasible = {
            tuple(int(bit) for bit in text)
            for text in metadata["feasible_patterns_logical_order"]
        }
        opt_hits = 0
        feasible_hits = 0
        mean_weight = 0.0
        for bitstring, value in counts.items():
            pattern = logical_pattern_from_qiskit(bitstring, metadata["n_qubits"])
            opt_hits += value * (pattern in optimal)
            feasible_hits += value * (pattern in feasible)
            mean_weight += value * sum(pattern) / shots
        empirical = counts_to_probabilities(counts, metadata["n_qubits"])
        ideal = np.asarray(metadata["ideal_probabilities_qiskit_index"], dtype=float)
        tv_distance = 0.5 * float(np.abs(empirical - ideal).sum())
        classical_fidelity = float(np.sqrt(empirical * ideal).sum() ** 2)
        block_results.append(
            {
                "tag": metadata["tag"],
                "original_qubits": metadata["original_qubits"],
                "shots": shots,
                "optimal_hits": opt_hits,
                "feasible_hits": feasible_hits,
                "p_opt": opt_hits / shots,
                "p_feasible": feasible_hits / shots,
                "p_opt_ideal": metadata["p_opt_ideal"],
                "p_opt_over_ideal": (opt_hits / shots) / metadata["p_opt_ideal"],
                "mean_local_weight": mean_weight,
                "total_variation_from_ideal": tv_distance,
                "classical_fidelity_to_ideal": classical_fidelity,
                "counts": counts,
            }
        )

    p_opt_product = float(math.prod(row["p_opt"] for row in block_results))
    p_feasible_product = float(math.prod(row["p_feasible"] for row in block_results))
    expected_weight = float(sum(row["mean_local_weight"] for row in block_results))

    rng = np.random.default_rng(73421)
    posterior_products = np.ones(posterior_draws, dtype=float)
    for row in block_results:
        posterior_products *= rng.beta(
            row["optimal_hits"] + 0.5,
            row["shots"] - row["optimal_hits"] + 0.5,
            size=posterior_draws,
        )
    full_draws = rng.beta(
        REFERENCE["full_optimal_hits"] + 0.5,
        REFERENCE["full_shots"] - REFERENCE["full_optimal_hits"] + 0.5,
        size=posterior_draws,
    )
    ci_low, posterior_median, ci_high = np.quantile(
        posterior_products, [0.025, 0.5, 0.975]
    )
    summary = {
        "p_opt_product_point_estimate": p_opt_product,
        "p_opt_product_posterior_median": float(posterior_median),
        "p_opt_product_95pct_credible_interval": [float(ci_low), float(ci_high)],
        "expected_optimal_hits_per_100k": 100_000 * p_opt_product,
        "expected_optimal_hits_per_100k_95pct_interval": [
            100_000 * float(ci_low),
            100_000 * float(ci_high),
        ],
        "p_feasible_product_point_estimate": p_feasible_product,
        "expected_global_cardinality": expected_weight,
        "fraction_of_ideal_p_opt": p_opt_product / REFERENCE["ideal_p_opt"],
        "ratio_to_full_hardware_p_opt": p_opt_product / REFERENCE["full_p_opt"],
        "posterior_probability_separate_better_than_full": float(
            np.mean(posterior_products > full_draws)
        ),
        "posterior_probability_separate_at_least_ideal": float(
            np.mean(posterior_products >= REFERENCE["ideal_p_opt"])
        ),
    }
    ideal_hits = REFERENCE["ideal_expected_hits_at_100k"]
    full_hits = REFERENCE["full_optimal_hits"]
    separate_hits = summary["expected_optimal_hits_per_100k"]
    summary.update(
        loss_from_ideal_pct=100.0 * (ideal_hits - separate_hits) / ideal_hits,
        improvement_over_full_pct=100.0 * (separate_hits - full_hits) / full_hits,
        fraction_of_ideal_to_full_gap_recovered=(
            (separate_hits - full_hits) / (ideal_hits - full_hits)
        ),
    )
    return {
        "schema_version": 1,
        "analyzed_utc": utc_now(),
        "job_id": plan.get("job_id"),
        "backend": plan["backend"],
        "method": (
            "produto das frequências de padrões ótimos dos 13 circuitos independentes; "
            "intervalo por posterior Jeffreys Beta(1/2,1/2), 200000 amostras"
        ),
        "reference": REFERENCE,
        "summary": summary,
        "blocks": block_results,
    }


def markdown_report(plan: dict[str, Any], analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    low, high = summary["expected_optimal_hits_per_100k_95pct_interval"]
    is_hardware = bool(analysis.get("job_id"))
    title = (
        "# Controle no IBM Kingston: 13 blocos separados"
        if is_hardware
        else "# Validação local ideal do controle por 13 blocos"
    )
    configuration = (
        "- Mitigação/configuração aplicada: DD XY4 e twirling de portas e medidas."
        if is_hardware
        else "- Configuração prevista para o hardware: DD XY4 e twirling de portas e medidas."
    )
    lines = [
        title,
        "",
        f"- Job: `{analysis.get('job_id') or 'simulação local'}`",
        f"- Backend: `{analysis['backend']}`",
        f"- Shots por bloco: {plan['shots_per_block']:,}",
        f"- Execuções de circuitos no total: {plan['total_circuit_shots']:,}",
        "- Ângulos preservados: $\\gamma=-0.04$ e $\\beta=0.35$.",
        configuration,
        "",
        "## Resultado principal",
        "",
        (
            f"O produto das probabilidades ótimas locais foi "
            f"${summary['p_opt_product_point_estimate']:.8g}$. Isso corresponde a "
            f"**{summary['expected_optimal_hits_per_100k']:.2f} hits ótimos por 100.000** "
            f"(intervalo de 95%: {low:.2f} a {high:.2f})."
        ),
        "",
        f"Referências: circuito ideal = {REFERENCE['ideal_expected_hits_at_100k']:.2f}/100.000; "
        f"circuito físico conjunto de 24 qubits = {REFERENCE['full_optimal_hits']}/100.000.",
        "",
        f"A execução separada reteve {100 * summary['fraction_of_ideal_p_opt']:.1f}% da "
        f"probabilidade ideal e foi {summary['ratio_to_full_hardware_p_opt']:.2f} vezes a "
        "probabilidade observada no circuito conjunto.",
        "",
        f"Ela recuperou {100 * summary['fraction_of_ideal_to_full_gap_recovered']:.1f}% "
        "da lacuna entre o circuito ideal e o circuito físico conjunto. Ainda permaneceu "
        f"{summary['loss_from_ideal_pct']:.1f}% abaixo do ideal.",
        "",
        "## Resultado por bloco",
        "",
        "| bloco | qubits originais | hits/shots | $P_i$ medido | $P_i$ ideal | razão | TV ideal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["blocks"]:
        lines.append(
            f"| {row['tag']} | {row['original_qubits']} | "
            f"{row['optimal_hits']}/{row['shots']} | {row['p_opt']:.6f} | "
            f"{row['p_opt_ideal']:.6f} | {row['p_opt_over_ideal']:.3f} | "
            f"{row['total_variation_from_ideal']:.3f} |"
        )
    hardware = analysis.get("hardware_job")
    if hardware:
        resources = hardware["summary"]
        lines += [
            "",
            "## Realização física",
            "",
            f"- Tempo quântico informado pela IBM: {hardware.get('quantum_seconds', float('nan')):.3f} s.",
            f"- Portas físicas de dois qubits somadas nos 13 circuitos: {resources['total_two_qubit_gates']}.",
            f"- Maior profundidade física individual: {resources['maximum_depth']}.",
            f"- Qubits físicos distintos reutilizados entre os circuitos: "
            f"{len(resources['distinct_physical_qubits'])} "
            f"({resources['distinct_physical_qubits']}).",
            f"- Produto do proxy independente porta × leitura: {resources['independent_error_proxy_product']:.4f}; "
            f"ele corresponderia a {resources['proxy_expected_hits_per_100k']:.2f} hits por 100.000.",
            "",
            "| bloco | qubits físicos | 2Q físicas | profundidade | proxy porta×leitura |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in hardware["blocks"]:
            lines.append(
                f"| {row['tag']} | {row['measured_physical_qubits']} | "
                f"{row['n_two_qubit_gates']} | {row['depth']} | {row['F_exact']:.4f} |"
            )
    lines += [
        "",
        "## Interpretação",
        "",
        "Este número é uma reconstrução estatística. Os 13 blocos foram executados como "
        "circuitos independentes e o sucesso global foi calculado por",
        "",
        "$$",
        "P_{\\mathrm{sep}}(\\mathrm{ótimo})=\\prod_{i=1}^{13}P_i(O_i).",
        "$$",
        "",
        "A igualdade usa a fatoração exata das 288 soluções ótimas em conjuntos locais. "
        "Ela não afirma que houve uma única medição simultânea dos 24 qubits. Se o produto "
        "se aproximar do ideal e superar claramente 79/100.000, a perda do circuito conjunto "
        "é compatível com o custo físico de mapear/executar todos os blocos de uma vez "
        "(mais qubits lidos, roteamento e ruído acumulado). Se o produto também ficar baixo, "
        "parte relevante da perda já ocorre dentro dos pequenos blocos.",
        "",
    ]
    if hardware:
        resources = hardware["summary"]
        lines += [
            "Neste job, o resultado separado ficou entre o circuito conjunto e o ideal. "
            "Portanto, há perdas mensuráveis dentro dos blocos, mas elas não explicam toda "
            "a queda para 79/100.000. O circuito conjunto tinha 49 portas físicas de dois "
            "qubits e profundidade 64; os blocos somaram 51 portas, e o maior deles teve "
            "profundidade 63. A melhora não pode ser atribuída simplesmente a menos portas. "
            "Ela é consistente com reutilizar poucos qubits físicos bem calibrados e evitar "
            "execução simultânea/crosstalk entre os blocos.",
            "",
            "Essa comparação não isola uma causa única: os dois jobs ocorreram em datas "
            "diferentes e receberam mapeamentos físicos diferentes. Assim, o resultado é "
            "evidência de uma penalidade da realização física conjunta, não uma medição "
            "exclusiva de crosstalk.",
            "",
        ]
    if not is_hardware:
        lines += [
            "> Este arquivo contém somente amostragem multinomial da distribuição ideal. "
            "Ele valida o programa, mas não é um resultado do IBM Kingston.",
            "",
        ]
    return "\n".join(lines)


def extract_runtime_data(job_id: str, plan: dict[str, Any]):
    job = runtime_service().job(job_id)
    print(f"job {job_id}: {job.status()}")
    result = job.result()
    counts = [pub.data.meas.get_counts() for pub in result]
    metadata: dict[str, Any] = {
        "job_id": job_id,
        "status": str(job.status()),
        "creation_date": str(job.creation_date),
        "quantum_seconds": float(job.usage_estimation.get("quantum_seconds", float("nan"))),
        "blocks": [],
    }
    try:
        properties = job.properties()
        pubs = job.inputs["pubs"]
        metadata["calibration_date"] = str(properties.last_update_date)
        edge_error: dict[tuple[int, int], float] = {}
        for gate in properties.gates:
            if len(gate.qubits) != 2:
                continue
            values = [parameter.value for parameter in gate.parameters if parameter.name == "gate_error"]
            if values:
                edge_error[tuple(sorted(gate.qubits))] = float(values[0])
        for block, pub in zip(plan["blocks"], pubs):
            circuit = pub[0]
            measured = measured_physical_qubits(circuit)
            pairs = [
                tuple(sorted(circuit.find_bit(q).index for q in instruction.qubits))
                for instruction in circuit.data
                if instruction.operation.num_qubits == 2
            ]
            multiplicity = Counter(pairs)
            missing = [edge for edge in multiplicity if edge not in edge_error]
            if missing:
                raise RuntimeError(f"calibração ausente para arestas {missing}")
            f_gates = float(
                np.prod([(1.0 - edge_error[edge]) ** count for edge, count in multiplicity.items()])
            )
            readout_errors = {q: float(properties.readout_error(q)) for q in measured}
            f_readout = float(np.prod([1.0 - error for error in readout_errors.values()]))
            metadata["blocks"].append(
                {
                    "tag": block["tag"],
                    "measured_physical_qubits": measured,
                    "n_two_qubit_gates": len(pairs),
                    "depth": circuit.depth(),
                    "two_qubit_edges": [
                        {"edge": list(edge), "count": count, "error": edge_error[edge]}
                        for edge, count in sorted(multiplicity.items())
                    ],
                    "readout_errors": readout_errors,
                    "F_gates": f_gates,
                    "F_readout": f_readout,
                    "F_exact": f_gates * f_readout,
                }
            )
        proxy = float(math.prod(row["F_exact"] for row in metadata["blocks"]))
        physical_qubits = sorted(
            {q for row in metadata["blocks"] for q in row["measured_physical_qubits"]}
        )
        metadata["summary"] = {
            "total_two_qubit_gates": sum(row["n_two_qubit_gates"] for row in metadata["blocks"]),
            "maximum_depth": max(row["depth"] for row in metadata["blocks"]),
            "distinct_physical_qubits": physical_qubits,
            "independent_error_proxy_product": proxy,
            "proxy_expected_hits_per_100k": (
                REFERENCE["ideal_expected_hits_at_100k"] * proxy
            ),
        }
    except Exception as exc:
        metadata["calibration_extraction_error"] = f"{type(exc).__name__}: {exc}"
    return counts, metadata


def command_analyze(args: argparse.Namespace) -> None:
    plan = load_json(args.plan)
    hardware_metadata = None
    if args.counts:
        payload = load_json(args.counts)
        count_list = payload.get("circuits", payload)
        if isinstance(count_list, dict):
            count_list = [count_list[row["tag"]] for row in plan["blocks"]]
        count_list = [entry.get("counts", entry) for entry in count_list]
    else:
        job_id = args.job_id or plan.get("job_id")
        if not job_id:
            raise SystemExit("nenhum job_id no plano; informe --job-id ou --counts")
        count_list, hardware_metadata = extract_runtime_data(job_id, plan)
        plan["job_id"] = job_id
    analysis = analyze_count_list(plan, count_list)
    if hardware_metadata:
        analysis["hardware_job"] = hardware_metadata
    write_json(args.out, analysis)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(plan, analysis))
    if hardware_metadata:
        plan["status"] = "analyzed"
        plan["analyzed_utc"] = utc_now()
        plan["analysis_path"] = str(args.out.resolve())
        plan["report_path"] = str(args.report.resolve())
        write_json(args.plan, plan)
    summary = analysis["summary"]
    low, high = summary["expected_optimal_hits_per_100k_95pct_interval"]
    print(f"P_sep(ótimo) = {summary['p_opt_product_point_estimate']:.9g}")
    print(
        f"hits equivalentes/100.000 = {summary['expected_optimal_hits_per_100k']:.3f} "
        f"(95%: {low:.3f}–{high:.3f})"
    )
    print(f"ideal = {REFERENCE['ideal_expected_hits_at_100k']:.3f}/100.000")
    print(f"24 qubits no Kingston = {REFERENCE['full_optimal_hits']}/100.000")
    print(f"análise JSON: {args.out}")
    print(f"relatório: {args.report}")


def command_simulate(args: argparse.Namespace) -> None:
    """Valida toda a análise com contagens multinomiais da distribuição ideal."""
    if args.plan == DEFAULT_PLAN:
        args.plan = VALIDATION_PLAN
    plan, _ = make_plan(args.instance, args.backend, args.shots)
    plan["status"] = "local_ideal_simulation"
    write_json(args.plan, plan)
    rng = np.random.default_rng(args.seed)
    circuits = []
    for row in plan["blocks"]:
        probabilities = np.asarray(row["ideal_probabilities_qiskit_index"], dtype=float)
        sampled = rng.multinomial(args.shots, probabilities / probabilities.sum())
        n = row["n_qubits"]
        counts = {
            format(index, f"0{n}b"): int(value)
            for index, value in enumerate(sampled)
            if value
        }
        circuits.append({"tag": row["tag"], "counts": counts})
    counts_path = args.plan.with_name("ideal_simulation_counts.json")
    write_json(counts_path, {"kind": "ideal_local_multinomial", "circuits": circuits})
    analysis = analyze_count_list(plan, [row["counts"] for row in circuits])
    write_json(args.out, analysis)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(plan, analysis))
    print_plan(plan)
    print(f"\nSimulação local salva em {counts_path}")
    print(f"Análise salva em {args.out}")
    print(f"Relatório salvo em {args.report}")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    common.add_argument("--backend", default=BACKEND)
    common.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)

    main = argparse.ArgumentParser(
        description="Controle do QAOA PEF-24 com 13 blocos separados no IBM Kingston"
    )
    sub = main.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", parents=[common], help="valida a fatoração e escreve o plano")
    submit = sub.add_parser("submit", parents=[common], help="transpila e submete ao IBM")
    submit.add_argument("--force-new", action="store_true")
    status = sub.add_parser("status", help="consulta um job sem baixar o resultado")
    status.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    status.add_argument("--job-id")
    analyze = sub.add_parser("analyze", help="baixa/analisa o job ou contagens arquivadas")
    analyze.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    analyze.add_argument("--job-id")
    analyze.add_argument("--counts", type=Path)
    analyze.add_argument("--out", type=Path, default=DEFAULT_ANALYSIS)
    analyze.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    simulate = sub.add_parser(
        "simulate", parents=[common], help="teste integral com amostras ideais locais"
    )
    simulate.add_argument("--seed", type=int, default=73421)
    simulate.add_argument("--out", type=Path, default=VALIDATION_ANALYSIS)
    simulate.add_argument("--report", type=Path, default=VALIDATION_REPORT)
    return main


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "shots") and args.shots <= 0:
        raise SystemExit("--shots deve ser positivo")
    commands = {
        "prepare": command_prepare,
        "submit": command_submit,
        "status": command_status,
        "analyze": command_analyze,
        "simulate": command_simulate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
