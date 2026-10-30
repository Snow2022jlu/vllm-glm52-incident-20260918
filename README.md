# vLLM GLM-5.2 KV 事故版本归档（2026-09-18）

完整 vLLM **0.28.0** 源码，包含事故镜像中已存在的四组定制补丁；Mooncake 固定为 **CUDA13 0.3.13.post1**。

| 补丁 | 用途 |
| --- | --- |
| P12 Prefill pressure | 真实 Prefill 压力统计，排除已完成请求，缺失样本不补零。 |
| isolated health/load | 独立于推理 API 的 EngineCore 负载与健康观测。 |
| compatible health/metrics | 原端口兼容健康与原生指标，推理流式转发。 |
| bounded health/metrics | API 心跳最多宽限 1800 秒，处理辅助进程故障与退出。 |

**这是包含已知风险的历史版本，未包含事故后 Mooncake 生命周期修复。** 四组补丁不能视为 KV 污染修复，也没有证据仅凭事故发生就归因于它们。

12 个定制运行文件及事故取证的 20 个源码文件均与历史 SHA 对应；根目录源码已应用补丁，无需重复应用。

- [版本、补丁、校验和构建说明](incident_archive/README.md)
- [版本与历史文件哈希](incident_archive/manifest.json)
- [四份顺序补丁](incident_archive/patches/)
- [最终 publisher](incident_archive/runtime/static_dp_registrar.py)
- [本次归档验证结果](incident_archive/validation.json)
- [上游原始 README](README.upstream.md)

```bash
uv run --no-project --python 3.12 incident_archive/verify.py --replay
```

仓库不包含生产凭据、内部服务绑定、现场日志、模型权重或二进制镜像。上游许可与版权声明保留。
