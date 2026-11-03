"""Predeclared plateau checkpoint; scheduler horizon stays unchanged."""


def plateau_stop(history, best_epoch, policy, horizon):
    if not policy:
        return False
    if policy.get('mode') == 'metric_decline':
        check = int(policy['check_epoch'])
        delta = float(policy.get('min_delta', 0.0))
        if not 2 <= check <= horizon or not 0 <= delta < 1:
            raise ValueError('Invalid metric-decline boundary')
        if len(history) != check:
            return False
        import math
        metric = policy.get('metric', 'pseudo')
        values = [row.get('metrics', {}).get(metric, {}).get('macro_auc') for row in history]
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError('Missing/nonfinite metric at early-stop boundary')
        # A concrete downward step at the agreed boundary, not a lower train loss.
        return values[-1] < values[-2] - delta
    if policy.get('mode', 'best_epoch') != 'best_epoch':
        raise ValueError('Unknown early-stop mode')
    check, reference = int(policy['check_epoch']), int(policy['reference_epoch'])
    if not 1 <= reference < check <= horizon:
        raise ValueError('Invalid predeclared early-stop boundary')
    return len(history) == check and 0 <= best_epoch < reference
