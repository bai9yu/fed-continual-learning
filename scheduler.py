"""

================================================================================

客户端调度

================================================================================

基线名（不含 _TrueLabel / _FullTrueLabel 后缀）：

    - NoClientTrain:  无客户端训练

    - AllClientsTrain: 所有无标签客户端参与

    - Random: 随机；若开启 use_wireless_scheduling 则 shuffle 后按上行带宽贪心接入

    - BestChannel: 按信道增益降序，再带宽贪心（关闭无线时取前 clients_per_round）

    - NewClassClientsOnly: 仅从 config['continual_new_class_client_ids']（默认 18..29）中选客户端；

      与 Random 相同但候选池受限

================================================================================

"""



import random

from typing import List, Optional, Dict, Tuple

import numpy as np


from modules.flfl import FLFL_STRATEGY_OVERRIDES
from modules.fedlgmatch import FEDLGMATCH_STRATEGY_OVERRIDES
from modules.fssl_uc import FSSL_UC_STRATEGY_OVERRIDES





def _new_class_client_pool(unlabeled_pool: List[int], config: Optional[Dict]) -> List[int]:

    """无标签池与「含新类客户端 id」配置的交集，保持 unlabeled_pool 内顺序。"""

    cfg = config or {}

    ids = cfg.get("continual_new_class_client_ids")

    if ids is None:

        ids = list(range(18, 30))

    else:

        ids = list(ids)

    allow = set(ids)

    return [cid for cid in unlabeled_pool if cid in allow]





def _bandwidth_greedy(

    ordered_client_ids: List[int],

    wireless_channel,

    channel_gains: Dict[int, float],

    comp_delays: Dict[int, float],

) -> List[int]:

    from modules.wireless_channel import greedy_select_clients_by_bandwidth



    selected, _ = greedy_select_clients_by_bandwidth(

        ordered_client_ids, wireless_channel, channel_gains, comp_delays

    )

    return selected





FULL_TRUE_LABEL_SUFFIX = "_FullTrueLabel"

MASKED_TRUE_LABEL_SUFFIX = "_TrueLabel"


# R_PO + 无线侧「伪标签快照」排序所用的 priority_old_new（与 federated.py 在每轮开头生成 mask 的时机对齐）
_WIFI_PRIO_PASS_FRAC_PASS_MEAN_CONF = "pass_frac_plus_pass_mean_conf"
_WIFI_PRIO_PASS_FRAC_LAM_NEW_THR_MEAN_CONF = "pass_frac_plus_lam_mean_conf_pred_new_on_pass"
_WIFI_PRIO_PASS_MEAN_CONF_LAM_NEW_THR_FRAC = "pass_mean_conf_plus_lam_frac_pred_new_on_pass"


def _r_po_wifi_sched_pass_features(
    client_dataset,
    k_old: int,
) -> Tuple[float, float, float, float]:
    """
    返回 (ρ_pass, mean_conf_on_pass, mean_conf_pass_and_pred_new, frac_pred_new_given_pass)。
    「新类」由伪标签类 id >= frame1_initial_num_classes (k_old) 定义。
    """
    u = len(client_dataset)
    if u <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if client_dataset.pseudo_mask is None:
        return 0.0, 0.0, 0.0, 0.0
    m = np.asarray(client_dataset.pseudo_mask, dtype=bool)
    n = int(m.sum())
    rho_pass = n / float(u)
    if n == 0:
        return rho_pass, 0.0, 0.0, 0.0
    conf = np.asarray(client_dataset.pseudo_confidence, dtype=np.float64)
    pl = np.asarray(client_dataset.pseudo_labels, dtype=np.int64)
    mu_pass = float(conf[m].mean())
    if int(k_old) <= 0:
        pred_new = np.zeros_like(m, dtype=bool)
    else:
        pred_new = m & (pl >= int(k_old))
    n_pn = int(pred_new.sum())
    frac_pn = n_pn / float(n)
    mu_pn = float(conf[pred_new].mean()) if n_pn > 0 else 0.0
    return rho_pass, mu_pass, mu_pn, frac_pn


def _r_po_wifi_sched_score(
    mode: str,
    rho_pass: float,
    mean_conf_pass: float,
    mean_conf_pred_new_on_pass: float,
    frac_pred_new_on_pass: float,
    lam: float,
    eps: float,
) -> float:
    if mode == _WIFI_PRIO_PASS_FRAC_PASS_MEAN_CONF:
        return rho_pass + mean_conf_pass + eps
    if mode == _WIFI_PRIO_PASS_FRAC_LAM_NEW_THR_MEAN_CONF:
        return rho_pass + lam * mean_conf_pred_new_on_pass + eps
    if mode == _WIFI_PRIO_PASS_MEAN_CONF_LAM_NEW_THR_FRAC:
        return mean_conf_pass + lam * frac_pred_new_on_pass + eps
    raise ValueError(f"未知的 wireless_client_priority_mode: {mode!r}")


def _sort_pool_by_r_po_wifi_scores(
    pool: List[int],
    data_loader,
    cfg: Dict,
    mode: str,
    channel_gains: Optional[Dict[int, float]] = None,
) -> List[int]:
    """按调度分降序；同分按上行信道增益降序（若提供）；再按 client_id 升序。"""
    k_old = int(cfg.get("frame1_initial_num_classes", 0))
    lam = float(cfg.get("r_po_wireless_sched_lambda", 1.0))
    eps = float(cfg.get("r_po_wireless_sched_score_eps", 1e-8))
    scored: Dict[int, float] = {}
    for cid in pool:
        rho, mu_p, mu_pn, frac_pn = _r_po_wifi_sched_pass_features(
            data_loader.client_datasets[cid], k_old
        )
        scored[cid] = _r_po_wifi_sched_score(
            mode, rho, mu_p, mu_pn, frac_pn, lam, eps
        )
    return sorted(
        pool,
        key=lambda cid: (
            -scored[cid],
            -float(channel_gains[cid])
            if channel_gains is not None and cid in channel_gains
            else 0.0,
            cid,
        ),
    )


# 策略型实验名（调度上等价 Random，伪标签策略由实验名决定；见 STRATEGY_EXPERIMENT_OVERRIDES）

STRATEGY_EXPERIMENT_OVERRIDES = {

    "Random_PriorityOldNew": {

        "pseudo_label_policy": "priority_old_new",

    },

    "PO_FPM": {
        "pseudo_label_policy": "priority_old_new",
        "wireless_client_priority_mode": _WIFI_PRIO_PASS_FRAC_PASS_MEAN_CONF,
    },

    "PO_FPLN": {
        "pseudo_label_policy": "priority_old_new",
        "wireless_client_priority_mode": _WIFI_PRIO_PASS_FRAC_LAM_NEW_THR_MEAN_CONF,
    },

    "PO_MFLN": {
        "pseudo_label_policy": "priority_old_new",
        "wireless_client_priority_mode": _WIFI_PRIO_PASS_MEAN_CONF_LAM_NEW_THR_FRAC,
    },

    "Random_StickyNew": {

        "pseudo_label_policy": "sticky_new",

    },

    "Random_FLFL": dict(FLFL_STRATEGY_OVERRIDES),

    "Random_FedLGMatch": dict(FEDLGMATCH_STRATEGY_OVERRIDES),

    "Random_FSSL_UC": dict(FSSL_UC_STRATEGY_OVERRIDES),

}





def parse_client_supervision_suffix(experiment_type: str):

    """

    解析实验名后缀，得到基线调度名与客户端监督模式（调度仅依赖基线名）。



    返回:

        (基线实验类型, supervision_mode)，其中 supervision_mode 为:

        - ``pseudo``: 默认，过阈值 + 伪类别监督

        - ``masked_true``: ``*_TrueLabel``，过阈值 + 真实类别监督（与伪标签同 mask）

        - ``full_true``: ``*_FullTrueLabel``，全样本 + 真实类别监督（不生成伪标签）

    """

    if experiment_type.endswith(FULL_TRUE_LABEL_SUFFIX):

        return experiment_type[: -len(FULL_TRUE_LABEL_SUFFIX)], "full_true"

    if experiment_type.endswith(MASKED_TRUE_LABEL_SUFFIX):

        return experiment_type[: -len(MASKED_TRUE_LABEL_SUFFIX)], "masked_true"

    return experiment_type, "pseudo"





def get_experiment_strategy_overrides(experiment_type: str) -> Dict:

    """去掉 TrueLabel/FullTrueLabel 后缀后，若为名策略实验则返回与 DEFAULT 合并用的字典。"""

    base, _ = parse_client_supervision_suffix(experiment_type)

    ovr = STRATEGY_EXPERIMENT_OVERRIDES.get(base)

    return dict(ovr) if ovr else {}





def select_clients_by_experiment(

    experiment_type: str,

    round_num: int,

    data_loader,

    clients_per_round: int = 5,

    config: Optional[Dict] = None,

    wireless_channel=None,

    channel_gains: Optional[Dict[int, float]] = None,

    comp_delays: Optional[Dict[int, float]] = None,

    use_wireless_scheduling: bool = False,

) -> List[int]:

    """

    根据实验类型选择客户端。



    - NoClientTrain / AllClientsTrain：行为不变（不参与带宽约束）。

    - Random / BestChannel / NewClassClientsOnly：若 ``use_wireless_scheduling`` 为 True 且传入信道与延迟，

      则先得到有序队列，再按上行带宽 best-effort 贪心接入。

    - 后缀 ``_TrueLabel`` / ``_FullTrueLabel`` 仅影响客户端本地监督方式，调度与去掉后缀后的基线相同。

    - ``Random_FedLGMatch`` / ``Random_FSSL_UC``：调度同 Random。

    - ``PO_FPM`` / ``PO_FPLN`` / ``PO_MFLN``：与 R_PO 相同伪标签；
      开启无线时在带宽贪心前先按各自 ``wireless_client_priority_mode`` 排序，同分时按信道增益次之。

    """

    base_raw, _ = parse_client_supervision_suffix(experiment_type)

    overrides = STRATEGY_EXPERIMENT_OVERRIDES.get(base_raw, {})

    sched_type = "Random" if base_raw in STRATEGY_EXPERIMENT_OVERRIDES else base_raw

    config_eff = {**(config or {}), **overrides}



    num_total = data_loader.num_clients

    unlabeled_pool = list(range(num_total))



    if sched_type == "NoClientTrain":

        return []



    if sched_type == "AllClientsTrain":

        return unlabeled_pool



    use_ws = (

        use_wireless_scheduling

        and wireless_channel is not None

        and channel_gains is not None

        and comp_delays is not None

    )

    base_seed = config_eff.get("seed", 42)



    def top_k_from_order(order: List[int]) -> List[int]:

        k = min(clients_per_round, len(order))

        return order[:k]



    if sched_type == "BestChannel":

        if channel_gains is None:

            raise ValueError("BestChannel 需要 channel_gains（由 WirelessChannel 每轮计算）")

        order = sorted(

            unlabeled_pool,

            key=lambda cid: float(channel_gains.get(cid, 0.0)),

            reverse=True,

        )

        if use_ws:

            return _bandwidth_greedy(order, wireless_channel, channel_gains, comp_delays)

        return top_k_from_order(order)



    if sched_type == "NewClassClientsOnly":

        pool = _new_class_client_pool(unlabeled_pool, config_eff)

        if not pool:

            raise ValueError(

                "NewClassClientsOnly：候选客户端为空。请检查 continual_new_class_client_ids 是否与无标签客户端 id 有交集。"

            )

        if use_ws:

            random.seed(base_seed + round_num * 1000)

            order = pool.copy()

            random.shuffle(order)

            return _bandwidth_greedy(order, wireless_channel, channel_gains, comp_delays)

        return random.sample(

            pool, k=min(clients_per_round, len(pool))

        )



    if use_ws:

        if sched_type == "Random":

            mode = overrides.get("wireless_client_priority_mode")

            if mode is None:

                random.seed(base_seed + round_num * 1000)

                order = unlabeled_pool.copy()

                random.shuffle(order)

                return _bandwidth_greedy(order, wireless_channel, channel_gains, comp_delays)

            order = _sort_pool_by_r_po_wifi_scores(
                unlabeled_pool, data_loader, config_eff, mode, channel_gains
            )

            return _bandwidth_greedy(order, wireless_channel, channel_gains, comp_delays)

        raise ValueError(f"未知的实验类型: {sched_type}")



    if sched_type == "Random":

        mode = overrides.get("wireless_client_priority_mode")

        random.seed(base_seed + round_num * 1001)

        if mode is None:

            return random.sample(

                unlabeled_pool, k=min(clients_per_round, len(unlabeled_pool))

            )

        order = _sort_pool_by_r_po_wifi_scores(
            unlabeled_pool, data_loader, config_eff, mode, None
        )

        return top_k_from_order(order)



    raise ValueError(f"未知的实验类型: {sched_type}")


