# -*- coding: utf-8 -*-
"""RequirementParser：需求文本 → StructuredRequirement（arch §2.3.6）。

B v0 纯规则实现（关键词词典 + 单位数值抽取），无 LLM、无训练参数。
LLM 前置解析属于 B v1（arch §2.3.9(b) 四层数据方案），本模块与其接口兼容。

抽取策略：
1. 属性：中英关键词词典命中（可多条 → 组合需求）；
2. 方向：不低于/至少 → bound_below；不超过/以内 → bound_above；缺省 bound_below；
3. 数值目标：正则抽 (数值, 单位) 对，按文本距离分配给最近的属性提及，
   单位归一化并按属主属性映射到标准键（kN/m→N/m，GPa→Pa，ms→s 等）；
4. 上下文规则："标准差/sigma_req" 在无干扰（鲁棒）语境时归 precision，
   有干扰语境时它是鲁棒规则里的信号方向约束，不单独成 precision 需求；
5. 二义性：R-COND 型文本默认读向 (a) 热锋快，细结构/保真/图样关键词 → 读向 (b)；
   两种情况都写 ambiguity_flags，禁止静默选择（arch §2.3.8 第 2 条）；
6. 无属性命中 → unmapped=True（R-NOMAP-000 流程入口）。
"""

from __future__ import annotations

import re

from b_network.metric_spec import StructuredRequirement

__all__ = ["RequirementParser"]

# 属性关键词（zh 在前 en 在后）
PROPERTY_KEYWORDS: dict[str, list[str]] = {
    "stiffness": ["刚度", "弹性模量", "硬度", r"\bstiffness\b", r"\bmodulus\b"],
    "precision": ["精密", "精度", "分辨", "可重复", "不确定度", "误差",
                  "定位", "方位估计",
                  r"\bprecision\b", r"\bresolve\b", r"\buncertaint",
                  r"\berror\b", r"\blocaliz", r"\baccurac"],
    "response_speed": ["响应", "弛豫", "恢复时间", "建立时间", "回到设定值", "回稳",
                       "压回", "回路稳定", "调频", "轨迹",
                       r"\bresponse\s*time\b", r"\bsettling\s*time\b",
                       r"\brecover", r"\bsettle\b", r"\bslew\b"],
    "transport": ["导热", "传热", "散热", "输运", "隔热", "热锋", "热点", "拉平",
                   "水锋", "湿润锋", "锋面", "扩散", "穿透", "覆盖", "降温",
                   "通风", "灌溉",
                   r"\bthermal\b", r"\bconductivity\b", r"\bheat\s*front\b",
                   r"\bmoisture\s*front\b", r"\birrigation\b", r"\btraverse\b",
                   r"\bfront\b", r"\bspread", r"\bpenetrat"],
    "robustness": ["鲁棒", "抗噪", "抗干扰", "抗振", "温漂", "干扰下", "免疫",
                   "结焦", "老化",
                   r"\brobust", r"\bnoise\b", r"\bfading\b", r"\bdrift\b",
                   r"\bvibration\b", r"\bwithout\s+drifting\b"],
    "stability": ["稳定", "相变", "退磁", "溶解线", "熔化点", "裕度", "保持窗口",
                  "工作点",
                  r"\bstab", r"\bbarrier\b", r"\btransition\b", r"\bmargin\b",
                  r"\bhold\b"],
    "adaptability": ["适应", "自适应", "多工况", "多环境", "重标定",
                     r"\badapt", r"\bterrain\b", r"\benvironment", r"\bcope\b"],
    "toughness": ["韧性", "断裂", "耐冲击", "吸能", "塌缩", "缓冲",
                  r"\btough", r"\bfracture\b", r"\bimpact\s*energy\b",
                  r"\babsorb", r"\bcrash\b", r"\bcollapse\b"],
}
# 单独"硬"字仅在无其他 stiffness 词命中时启用（避免与"硬度"重复计位）；
# 单独"锋"字同理（冷锋/膜通量锋/热锋…：仅在无其他 transport 词命中时启用，
# 避免与"热锋/水锋/扩散锋/锋面"重复计位）——2026-10-02 批次 3 补
SOFT_SINGLE_KW = {"stiffness": "硬", "transport": "锋"}

# 标准差/sigma_req：上下文敏感（无鲁棒语境才归 precision）
SIGMA_CONTEXT_KW = ["标准差", r"sigma_req", r"\bsigma\b", "不确定度", "波动",
                    r"\buncertaint"]

# R-COND 读向 (b) 触发词
FIDELITY_HINTS = ["细结构", "保真", "图样", "模糊", "细节", r"\bfidelity\b", r"\bfine[-\s]?structure\b"]

DIR_BELOW = ["不低于", "不小于", "至少", "不少于", "大于等于", ">=", "≥",
             r"\bat\s+least\b", r"\bno\s+less\s+than\b", r"\bmore\s+than\b"]
DIR_ABOVE = ["不超过", "不高于", "不大于", "小于", "少于", "以内", "<=", "≤",
             "至多", "压回", "回稳到",
             r"\bwithin\b", r"\bat\s+most\b", r"\bup\s+to\b",
             r"\bno\s+more\s+than\b", r"\bunder\b"]

# 单位正则（数值之后）——2026-10-02 口径统一扩表：刚度/模量/热扩散率/面积比
_UNIT_ALT = (r"kN/m|N/m|GPa|MPa|Pa|m2/s|m²/s|m\^2/s|ns|ms|毫秒|微秒|μs|us"
             r"|nats|pH|%|％|mrad|rad|弧度|deg|°|K|开尔文|°C|摄氏度"
             r"|mm|毫米|μm|um|米|秒|s|m")
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*("
                          + _UNIT_ALT + r")(?![0-9A-Za-z/])")

# 建立带（settling_band）：±1% → pm_1_percent；95% 带内建立 → 95_percent
_BAND_PM_RE = re.compile(r"±\s*(\d+(?:\.\d+)?)\s*%")
_BAND_PCT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:带|建立|回稳|行程|响应|stroke|settling|band"
    r"|response)")
# 观测时间（transport 读向 b）：观测 N s
_T_OBS_RE = re.compile(r"(?:观测|观察|记录)\s*(\d+(?:\.\d+)?)\s*(?:s|秒)")
# 单次读数（英文/中文惯例）→ n_samples_count = 1
_SINGLE_READING_RE = re.compile(
    r"\b(?:single|one)[-\s](?:reading|measurement|sample|shot|pulse)\b"
    r"|一次读数|单次读数|单次测量|一次测量")
# sigma_req 赋值式：text 里直接给出 sigma_req = X（无量纲或带单位）
_SIGMA_REQ_RE = re.compile(
    r"sigma_req\s*[=＝]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"\s*(kN/m|N/m|GPa|MPa|Pa|mm|μm|um|m|K|pH|%|％)?")
# 科学计数法异体：×10⁻⁶ / x10^-6 / 10⁻⁶ → e 记法（位置近似，够用于属主归属）
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _normalize_sci(text: str) -> str:
    t = text.translate(_SUP)
    t = re.sub(r"[×x]\s*10\s*\^?\s*([-+]?\d+)", r"e\1", t)
    t = re.sub(r"\b10\s*\^\s*([-+]?\d+)", r"1e\1", t)
    return t

# 单位 → (换算到 SI, 属主属性 → 标准键)；缺省键用于无属主信息的兜底
UNIT_MAP: dict[str, tuple[float, dict[str, str], str]] = {
    "kN/m":  (1e3,  {"stiffness": "stiffness_N_per_m"}, "stiffness_N_per_m"),
    "N/m":   (1.0,  {"stiffness": "stiffness_N_per_m"}, "stiffness_N_per_m"),
    "GPa":   (1e9,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "MPa":   (1e6,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "Pa":    (1.0,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "ns":    (1e-9, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "ms":    (1e-3, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "毫秒":  (1e-3, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "μs":    (1e-6, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "us":    (1e-6, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "微秒":  (1e-6, {"response_speed": "tau_s", "transport": "front_time_s"}, "tau_s"),
    "s":     (1.0,  {"response_speed": "settling_time_s",
                      "transport": "front_time_s"}, "settling_time_s"),
    "秒":    (1.0,  {"response_speed": "settling_time_s",
                      "transport": "front_time_s"}, "settling_time_s"),
    "ms":    (1e-3, {"response_speed": "settling_time_s",
                     "transport": "front_time_s"}, "settling_time_s"),
    "毫秒":  (1e-3, {"response_speed": "settling_time_s",
                     "transport": "front_time_s"}, "settling_time_s"),
    # 口径统一新增（2026-10-02）：刚度/模量/热扩散率
    "N/m":   (1.0,  {"stiffness": "stiffness_N_per_m"}, "stiffness_N_per_m"),
    "kN/m":  (1e3,  {"stiffness": "stiffness_N_per_m"}, "stiffness_N_per_m"),
    "Pa":    (1.0,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "MPa":   (1e6,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "GPa":   (1e9,  {"stiffness": "modulus_Pa"}, "modulus_Pa"),
    "m2/s":  (1.0,  {"transport": "kappa_m2_per_s"}, "kappa_m2_per_s"),
    "m²/s":  (1.0,  {"transport": "kappa_m2_per_s"}, "kappa_m2_per_s"),
    "m^2/s": (1.0,  {"transport": "kappa_m2_per_s"}, "kappa_m2_per_s"),
    "K":     (1.0,  {"precision": "sigma_K", "stability": "T_c_K"}, "sigma_K"),
    "%":     (0.01, {"precision": "sigma_dimensionless_fraction",
                     "robustness": "sigma_dimensionless_fraction"},
              "sigma_dimensionless_fraction"),
    "％":    (0.01, {"precision": "sigma_dimensionless_fraction",
                     "robustness": "sigma_dimensionless_fraction"},
              "sigma_dimensionless_fraction"),
    "百分比": (0.01, {"precision": "sigma_dimensionless_fraction",
                      "robustness": "sigma_dimensionless_fraction"},
               "sigma_dimensionless_fraction"),
    "开尔文": (1.0, {"precision": "sigma_K"}, "sigma_K"),
    "°C":    (1.0,  {"precision": "sigma_K"}, "sigma_K"),
    "摄氏度": (1.0, {"precision": "sigma_K"}, "sigma_K"),
    "pH":    (1.0,  {"precision": "sigma_pH", "robustness": "sigma_pH"}, "sigma_pH"),
    # 角度单位（声学/光学方位类）：mrad/rad/deg → sigma_rad（弧度）
    "rad":   (1.0,   {"precision": "sigma_rad"}, "sigma_rad"),
    "弧度":  (1.0,   {"precision": "sigma_rad"}, "sigma_rad"),
    "mrad":  (1e-3,  {"precision": "sigma_rad"}, "sigma_rad"),
    "deg":   (0.017453292519943295, {"precision": "sigma_rad"}, "sigma_rad"),
    "°":     (0.017453292519943295, {"precision": "sigma_rad"}, "sigma_rad"),
    "mm":    (1e-3, {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "毫米":  (1e-3, {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "μm":    (1e-6, {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "um":    (1e-6, {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "微米":  (1e-6, {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "m":     (1.0,  {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "米":    (1.0,  {"transport": "length_scale_m", "precision": "sigma_m"}, "sigma_m"),
    "nats":  (1.0,  {"stability": "E_req_nats", "toughness": "E_req_nats"}, "E_req_nats"),
}

_N_SAMPLES_RE = re.compile(
    r"(\d+)\s*(?:次|个独立样本|个样本|个符合事件|个光子|个事件|个脉冲|个计数"
    r"|个采样点|采样点|个快拍|快拍|次读数|次平均"
    r"|samples|readings|pulses|events|counts|coincidences|snapshots"
    r"|sampling\s+points)")
_N_ENV_RE = re.compile(r"(\d+)\s*(?:种|个)\s*(?:工况|信道状态|环境|地形|塌缩模式|能量谷)")
# 数词支持（B 类残差补，2026-10-02）：中文数词 / 英文数词 + 英文量词
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10}
_EN_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
           "twelve": 12, "twenty": 20}
_N_ENV_CN_RE = re.compile(
    r"([0-9]+|[一二三四五六七八九十两])\s*(?:种|个)\s*"
    r"(?:[^\s，。;；]{0,6})?(工况|信道状态|环境|地形|塌缩模式|能量谷|谷|模式)")
_N_ENV_EN_RE = re.compile(
    r"([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|twenty)\s+(terrain|environment|valley|mode|gait|channel)"
    r"\s*(?:classes|class|states?|s)?", re.IGNORECASE)


def _numeral(tok: str) -> int | None:
    """数词 → 整数（阿拉伯数字 / 中文数词 / 英文数词）。"""
    if tok.isdigit():
        return int(tok)
    if tok in _CN_NUM:
        return _CN_NUM[tok]
    return _EN_NUM.get(tok.lower())


def _positions(text: str, patterns: list[str]) -> list[int]:
    out = []
    for pat in patterns:
        out.extend(m.start() for m in re.finditer(pat, text))
    return out


class RequirementParser:
    """规则版需求解析器（B v0，无训练参数）。"""

    def parse(self, text: str) -> StructuredRequirement:
        req = StructuredRequirement(requirement_text=text)

        # ---- 1. 属性命中 ----
        prop_positions: dict[str, list[int]] = {}
        for prop, kws in PROPERTY_KEYWORDS.items():
            pos = _positions(text, kws)
            if pos:
                prop_positions[prop] = pos
        for prop, kw in SOFT_SINGLE_KW.items():
            if prop not in prop_positions:
                pos = _positions(text, kw)
                if pos:
                    prop_positions[prop] = pos

        # ---- 1b. 标准差/sigma_req 上下文规则 ----
        sigma_pos = _positions(text, SIGMA_CONTEXT_KW)
        if sigma_pos and "robustness" not in prop_positions:
            prop_positions.setdefault("precision", []).extend(sigma_pos)

        req.property_ids = list(prop_positions)
        if not req.property_ids:
            req.unmapped = True
            req.parser_notes.append("无本体属性命中 → R-NOMAP-000 流程（禁止硬映射）")
            return req

        # ---- 2. 方向（正则匹配：正则模式与中文子串混排）----
        dir_hit = any(re.search(k, text) for k in DIR_ABOVE + DIR_BELOW)
        req.direction = ("bound_above"
                         if any(re.search(k, text) for k in DIR_ABOVE)
                         else "bound_below")
        # 隐式上界线索（v0.1，2026-10-02 口径统一）：
        #   a) sigma_req / 标准差 / 不确定度 + 数值 → σ 的上界约束（按定义）；
        #   b) 「在 N <时间单位> 内」→ 时间上界（「内」单独不构成关键词）。
        if not dir_hit:
            if re.search(r"(sigma_req|\bsigma\b|标准差|不确定度|波动)", text) \
                    and re.search(r"\d", text):
                req.direction = "bound_above"
                req.parser_notes.append("方向由隐式线索判定：sigma 类表述 → 上界")
            elif re.search(r"在\s*\d+(?:\.\d+)?\s*"
                           r"(?:ms|s|秒|毫秒|μs|us|h|小时|min|分钟|天|d)\s*内",
                           text):
                req.direction = "bound_above"
                req.parser_notes.append("方向由隐式线索判定：「在 N 时间内」→ 上界")
            elif "precision" in req.property_ids:
                # σ_req 按定义是上界：precision 语境无方向词 → 默认上界
                req.direction = "bound_above"
                req.parser_notes.append(
                    "方向由隐式线索判定：precision 语境无方向词 → σ_req 上界")

        # ---- 3. 数值目标（属主感知的键映射） ----
        text = _normalize_sci(text)          # 科学计数法异体归一（口径统一）
        prop_start = {p: min(ps) for p, ps in prop_positions.items()}
        for m in _NUM_UNIT_RE.finditer(text):
            value, unit = float(m.group(1)), m.group(2)
            info = UNIT_MAP.get(unit)
            if info is None:
                continue
            scale, by_prop, default_key = info
            owner = min(prop_start, key=lambda p: abs(prop_start[p] - m.start()))
            key = by_prop.get(owner, default_key)
            req.target_value.setdefault(key, round(value * scale, 12))

        for m in _N_SAMPLES_RE.finditer(text):
            req.target_value.setdefault("n_samples_count", int(m.group(1)))
        if _SINGLE_READING_RE.search(text):
            req.target_value.setdefault("n_samples_count", 1)
        msr = _SIGMA_REQ_RE.search(text)
        if msr and "precision" in req.property_ids:
            val = float(msr.group(1))
            unit = msr.group(2)
            if unit in ("%", "％"):
                req.target_value.setdefault("sigma_dimensionless_fraction",
                                            round(val / 100.0, 12))
            elif unit:
                key = UNIT_MAP.get(unit, (1.0, {}, "sigma_K"))[2]
                req.target_value.setdefault(key, val * UNIT_MAP[unit][0])
            else:
                req.target_value.setdefault("sigma_dimensionless_fraction", val)
        # 裸无量纲精度值（无量词/无单位）：「压到 0.02 以内」「压到 0.15」
        if "precision" in req.property_ids and not any(
                k.startswith("sigma") for k in req.target_value):
            mb = re.search(r"(?:压到|不超过|不大于|以内|小于)\s*"
                           r"(\d+(?:\.\d+)?)\s*(?![0-9A-Za-z%])", text)
            if mb:
                req.target_value["sigma_dimensionless_fraction"] = \
                    float(mb.group(1))
        for m in _N_ENV_RE.finditer(text):
            # 口径统一：工况/环境/信道状态 → n_environments_count；
            # 能量谷/塌缩模式 → n_valleys_count（两者皆为规范键）
            tok = m.group(0)
            if re.search(r"谷|塌缩", tok):
                req.target_value.setdefault("n_valleys_count", int(m.group(1)))
            else:
                req.target_value.setdefault("n_environments_count",
                                            int(m.group(1)))
        # 数词版（中文/英文）：三个能量谷 / 8 terrain classes
        for m in _N_ENV_CN_RE.finditer(text):
            val = _numeral(m.group(1))
            if val is None:
                continue
            # 分类看**整段匹配文本**（修饰词可能落在 group(2) 之外）
            if re.search(r"谷|塌缩", m.group(0)):
                req.target_value.setdefault("n_valleys_count", val)
            else:
                req.target_value.setdefault("n_environments_count", val)
        for m in _N_ENV_EN_RE.finditer(text):
            val = _numeral(m.group(1))
            if val is None:
                continue
            if re.search(r"valley", m.group(2), re.IGNORECASE):
                req.target_value.setdefault("n_valleys_count", val)
            else:
                req.target_value.setdefault("n_environments_count", val)
        # 建立带（规范性字符串键）
        mb = _BAND_PM_RE.search(text)
        if mb:
            req.target_value.setdefault(
                "settling_band", "pm_%s_percent" % mb.group(1).rstrip("0.")
                if "." not in mb.group(1) else
                "pm_%s_percent" % mb.group(1).replace(".", "_"))
        elif _BAND_PCT_RE.search(text):
            pct = _BAND_PCT_RE.search(text).group(1)
            req.target_value.setdefault("settling_band",
                                        "%s_percent" % pct.rstrip("0."))
            # 该百分数是建立带而非不确定度：撤掉主循环误收的 sigma 无量纲键
            if abs(float(req.target_value.get(
                    "sigma_dimensionless_fraction", -1))
                    - float(pct) / 100.0) < 1e-12:
                req.target_value.pop("sigma_dimensionless_fraction")
        # 观测时间（transport 读向 b）
        mt = _T_OBS_RE.search(text)
        if mt and "transport" in req.property_ids:
            req.target_value.setdefault("t_obs_s", float(mt.group(1)))
        # 工作温度语境（stability）：…工作温度 1080 K → T_work_K
        mw = re.search(r"(?:工作|运行)\s*(?:温度|点)?\s*(\d+(?:\.\d+)?)\s*K",
                       text)
        if mw and "stability" in req.property_ids:
            req.target_value["T_work_K"] = float(mw.group(1))
        # 稳定性无量纲裕度（% + stability）→ delta_dimensionless
        if "stability" in req.property_ids:
            mm = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if mm:
                req.target_value.setdefault(
                    "delta_dimensionless", round(float(mm.group(1)) / 100.0, 6))
            # 裸无量纲数（文本未带单位）：约化/裕度/距离 + 不低于/至少 0.15
            md = re.search(r"(?:约化|裕度|距离|margin)[^。;；]{0,12}?"
                           r"(\d+(?:\.\d+)?)\s*(?![0-9.%])", text)
            if md and "delta_dimensionless" not in req.target_value:
                req.target_value["delta_dimensionless"] = float(md.group(1))

        # reading(b) 保真读向：transport 的长度/时间键换为规范读向键
        # （口径统一：reading 决定键名——a: length_scale_m/front_time_s；
        #  b: ell_req_m/t_obs_s）
        if "transport" in req.property_ids and any(
                re.search(h, text) for h in FIDELITY_HINTS):
            if "length_scale_m" in req.target_value:
                req.target_value["ell_req_m"] = req.target_value.pop(
                    "length_scale_m")
            if "front_time_s" in req.target_value:
                req.target_value["t_obs_s"] = req.target_value.pop(
                    "front_time_s")

        # 响应速度语境的无 ± 百分比带：「回稳到设定带的 1% 以内」→ pm_1_percent
        if "response_speed" in req.property_ids and "settling_band" not in \
                req.target_value:
            mr = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if mr and re.search(r"带|band", text):
                v = mr.group(1).rstrip("0.") if "." in mr.group(1) else \
                    mr.group(1)
                req.target_value["settling_band"] = "pm_%s_percent" % (
                    v.replace(".", "_") if "." in mr.group(1) else v)
                if abs(float(req.target_value.get(
                        "sigma_dimensionless_fraction", -1))
                        - float(mr.group(1)) / 100.0) < 1e-12:
                    req.target_value.pop("sigma_dimensionless_fraction")

        # ---- 4. R-COND 二义性 ----
        if "transport" in req.property_ids:
            reading_b = bool(re.search("|".join(FIDELITY_HINTS), text))
            if reading_b:
                req.ambiguity_flags.append(
                    "R-COND-001 读向(b) 细结构保真（κ↓，q6 上界）：文本含保真类词，显式判读向 b")
            else:
                req.ambiguity_flags.append(
                    "R-COND-001 二义性：文本未显式区分读向；缺省取读向(a) 热锋传播快（κ↑，q7↑），"
                    "禁止静默选择（arch §2.3.8 第 2 条）")

        # ---- 5. RV 提示 ----
        if req.property_ids == ["stiffness"]:
            req.rv_hint = "microstate_gibbs"
        elif "response_speed" in req.property_ids:
            req.rv_hint = "markov_chain"
        elif "transport" in req.property_ids:
            req.rv_hint = "field_time_slices"

        if not req.target_value:
            req.parser_notes.append("未解析出数值目标：仅方向性匹配（数值抽取为 B v0 已知局限）")
        return req
