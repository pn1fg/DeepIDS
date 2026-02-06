# 特征定义文档

本项目使用固定的41维特征向量，覆盖协议、IP、端口、包大小与时间统计等信息。

## 特征列表

| 序号 | 特征名 | 含义 | 范围 |
| --- | --- | --- | --- |
| 1 | protocol_type | 协议类型编码(TCP=0, UDP=1, ICMP/OTHER=2) | 0-2 |
| 2 | src_ip_numeric | 源IP数值化 | 0-4294967295 |
| 3 | dst_ip_numeric | 目标IP数值化 | 0-4294967295 |
| 4 | src_port | 源端口 | 0-65535 |
| 5 | dst_port | 目标端口 | 0-65535 |
| 6 | packet_length | 数据包长度 | 0-9000 |
| 7 | inter_packet_time | 包间隔时间 | 0-10 |
| 8 | tcp_flags_syn | TCP SYN标志位 | 0-1 |
| 9 | tcp_flags_ack | TCP ACK标志位 | 0-1 |
| 10 | tcp_flags_fin | TCP FIN标志位 | 0-1 |
| 11 | tcp_flags_rst | TCP RST标志位 | 0-1 |
| 12 | tcp_flags_psh | TCP PSH标志位 | 0-1 |
| 13 | tcp_flags_urg | TCP URG标志位 | 0-1 |
| 14 | flow_direction | 流量方向(内外网) | 0-1 |
| 15 | packet_count | 包数量累计 | 0-1000000 |
| 16 | total_bytes | 字节总量累计 | 0-1000000000 |
| 17 | avg_packet_size | 平均包大小 | 0-9000 |
| 18 | max_packet_size | 最大包大小 | 0-9000 |
| 19 | min_packet_size | 最小包大小 | 0-9000 |
| 20 | std_packet_size | 包大小标准差 | 0-9000 |
| 21 | initial_window_size | 初始TCP窗口大小 | 0-65535 |
| 22 | avg_ttl | TTL平均值 | 0-255 |
| 23 | connection_duration | 连接持续时间 | 0-3600 |
| 24 | other_feature_1 | TCP窗口大小 | 0-65535 |
| 25 | other_feature_2 | TTL即时值 | 0-255 |
| 26 | other_feature_3 | TOS即时值 | 0-255 |
| 27 | other_feature_4 | IP层长度 | 0-65535 |
| 28 | other_feature_5 | 源IP是否私网 | 0-1 |
| 29 | other_feature_6 | 目标IP是否私网 | 0-1 |
| 30 | other_feature_7 | 源端口是否知名端口 | 0-1 |
| 31 | other_feature_8 | 目标端口是否知名端口 | 0-1 |
| 32 | other_feature_9 | 源端口是否临时端口 | 0-1 |
| 33 | other_feature_10 | 目标端口是否临时端口 | 0-1 |
| 34 | other_feature_11 | 包速率 | 0-100000 |
| 35 | other_feature_12 | 字节速率 | 0-10000000 |
| 36 | other_feature_13 | 包长度冗余 | 0-9000 |
| 37 | other_feature_14 | IP版本 | 0-6 |
| 38 | other_feature_15 | TCP标志位原始值 | 0-255 |
| 39 | other_feature_16 | TCP窗口平均值 | 0-65535 |
| 40 | other_feature_17 | 是否TCP | 0-1 |
| 41 | other_feature_18 | 是否UDP | 0-1 |

## 归一化说明

- 所有特征使用Min-Max归一化映射到[0, 1]
- 超出范围的值会被截断到配置边界
- 缺失值以0处理

## 使用示例

```python
from data.data_loader import PacketRecord
from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(config_path="config.yaml")
record = PacketRecord(
    timestamp=0.1,
    src_ip="10.0.0.1",
    dst_ip="10.0.0.2",
    protocol="TCP",
    src_port=1234,
    dst_port=80,
    length=128,
    tcp_flags=2,
    tcp_window=1024,
    ttl=64,
    tos=0,
    ip_length=128,
    icmp_type=None,
    icmp_code=None,
)
features = extractor.extract(record)
```
