class AnomalyMetrics:
    """Tracks client exclusion decisions against ground-truth malicious labels.

    Per-round, computes:
      - TP: malicious client correctly excluded
      - FP: benign client incorrectly excluded
      - TN: benign client correctly included
      - FN: malicious client incorrectly included (missed)
      - Precision, Recall, F1 derived from above
    """

    def __init__(self):
        self.rounds: list[dict] = []

    def compute_round(
        self,
        malicious_clients: set[int],
        excluded_clients: set[int],
        all_clients: set[int],
    ) -> dict:
        """Compute anomaly detection metrics for one round."""
        benign_clients = all_clients - malicious_clients

        tp = len(excluded_clients & malicious_clients)
        fp = len(excluded_clients & benign_clients)
        tn = len(benign_clients - excluded_clients)
        fn = len(malicious_clients - excluded_clients)

        # Precision: tp/(tp+fp) when exclusions exist; 0.0 when no exclusions but
        # malicious clients exist (missed all); NaN when nothing exists at all.
        # Recall: tp/(tp+fn) when malicious clients exist; NaN otherwise (undefined).
        precision = tp / (tp + fp) if (tp + fp) > 0 else (float('nan') if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        if precision != precision or recall != recall:  # NaN check
            f1 = float('nan')
        else:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "excluded": sorted(excluded_clients),
            "malicious": sorted(malicious_clients),
        }
        self.rounds.append(result)
        return result

    def summary(self) -> dict:
        """Return cumulative summary across all recorded rounds."""
        total_tp = sum(r["tp"] for r in self.rounds)
        total_fp = sum(r["fp"] for r in self.rounds)
        total_tn = sum(r["tn"] for r in self.rounds)
        total_fn = sum(r["fn"] for r in self.rounds)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else (float('nan') if total_tp + total_fn == 0 else 0.0)
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
        if precision != precision or recall != recall:  # NaN check
            f1 = float('nan')
        else:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "total_rounds": len(self.rounds),
            "cumulative_tp": total_tp,
            "cumulative_fp": total_fp,
            "cumulative_tn": total_tn,
            "cumulative_fn": total_fn,
            "cumulative_precision": precision,
            "cumulative_recall": recall,
            "cumulative_f1": f1,
        }
