"""B 网络组件包。

包含物理系统 → 信息结构图 → 指纹的编码管线（架构文档 §2.3.9(a) 的
ChannelTemplateLibrary 线）。完整 B 网络还包含 RuleRegistry（需求解析→规则匹配），
属需求映射线，另行实现。
"""

from .fingerprint_computer import FingerprintComputer
from .system_encoder import SystemEncoder

__all__ = ["SystemEncoder", "FingerprintComputer"]
