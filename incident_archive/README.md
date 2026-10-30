# 事故版本归档说明

这是 2026-09-18 GLM-5.2 共享 Mooncake L2 KV 串块事故所用 vLLM 定制版本的源码归档。
仓库根目录是完整 vLLM 源码，已经叠加历史镜像中确认存在的改动。

## 固定版本与来源

| 项目 | 版本或身份 |
| --- | --- |
| vLLM | 0.28.0 |
| 上游源码 commit | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| Mooncake 包 | `mooncake-transfer-engine-cuda13==0.3.13.post1` |
| Mooncake store.so SHA-256 | `3e477eee78de13fb121650fc896cb5d7576ec43a3c56ba59a6608c64f0abf6df` |
| 历史部署镜像 digest | `sha256:aa5b05979e0c3cc4a7f3ee59748dcb222ac773e2d8ad6a6c95bbe2ea64d96f96` |
| 数据传输 | P2PHANDSHAKE / RDMA |
| 历史 Master 参数 | client_ttl=300 秒、32 RPC 线程、free_ratio_first、高水位 0.85 |
| 历史 donor 容量 | 每个 2 TiB |

上述 Master/donor 配置仅为历史背景。本仓库不包含生产认证、节点绑定、集群 YAML、模型权重或现场日志。
Mooncake 原生代码和二进制通过包版本及哈希固定，未将后续开发分支混入此归档。

`manifest.json` 来自历史运行 provenance 中选出的版本和文件哈希。
12 个定制运行文件及事故取证的 20 个 Store/NIXL/KV 源码文件均逐一比对。
两组文件有重叠，不能相加当作 32 个不同文件。

## 四组补丁

`patches/` 的四份 diff 按编号应用到上述上游源码；根目录已经是全部应用后的状态，勿重复应用。

| 顺序 | 补丁 | 行为 |
| --- | --- | --- |
| 01 | P12 Prefill pressure | 统计未命中请求、未命中 token 和等待请求，排除已完成请求；配套采集器缺失样本不补零。 |
| 02 | isolated health/load | EngineCore 发布真实负载，独立进程提供观测，校验 rank、进程身份、epoch 和时效。 |
| 03 | compatible health/metrics | 原端口 health/metrics 由独立入口处理，保留原生累计指标，推理流量转发给内部 API；支持 degraded 状态。 |
| 04 | bounded health/metrics | API 心跳宽限上限 1800 秒，采集不续期；增加辅助进程故障检测、退出清理及 unavailable 处理。 |

最终 publisher 在 `runtime/static_dp_registrar.py`，SHA 与运行 provenance 一致。
`patch_kits/` 保留原始小型安装器、源码输入、payload、测试及历史元数据；其中的候选状态描述和源码清单属于制作补丁时的历史记录，不代表本次重新进行了 GPU 验收。
少量历史镜像引用仅用于来源核对，不附访问凭据；镜像检查辅助脚本依赖原环境，不能直接当作通用安装入口。

## 污染事故与后续修复的边界

事故中两个 COMPLETE 对象应分别对应 token 0–63 和 64–127，内容却对应 9664–9727 和 9728–9791，均错位 151 个 64-token 块。
旧版本允许逻辑传输失败返回时仍存在 pending 传输，调用方和远端 allocation 随后可能被复用。这是有源码依据的候选机制，历史首次污染的 writer、DMA 和地址代次关联仍未完整取证。

Store scheduler 原来已有保存 KV 时的 block 引用保护；这属于基线机制，不能额外算作本仓库四组定制补丁。
上述四组补丁主要处理观测与调度指标，不能称为已修复 KV 污染，也不能据事故发生推断这些补丁造成污染。

本归档不包含事故后 Mooncake `bd91ed8`、`0dc2e03`、`4745111`、`e1eb13d` 生命周期修复；更早 PR2044 定制 wheel 也没有被当作本版本依赖。
历史 provenance 中 `h02_fixed=false` 保留，不能声称健康状态已在所有 Gateway 路径得到完整处理。

## 校验与重放

在仓库根目录：

```bash
uv run --no-project --python 3.12 incident_archive/verify.py --replay
```

默认校验源码、publisher 和补丁 SHA 及 Python 3.12 语法。`--replay` 在临时目录从 `upstream-v0.28.0` 快照提取原文件，顺序应用四份补丁并核对最终运行文件；不修改当前源码。

可独立运行原 P12 的 26 项 CPU 回归：

```bash
uv run --no-project --python 3.12 --with prometheus-client \
  incident_archive/patch_kits/p12-prefill-pressure-vllm0280/test_candidate.py \
  --output-dir /tmp/p12-public-archive-new-run
```

输出目录必须尚不存在。本次上传校验见 `validation.json`。CPU 测试和哈希匹配不能代替完整 GPU、RDMA、模型输出或性能验收。

## 构建方式

使用固定官方镜像作为原生库来源，安装固定 Mooncake wheel，并覆盖 12 个已校验 Python 文件：

```bash
docker build -f incident_archive/Dockerfile -t vllm-glm52-incident:20260918 .
```

这个 Dockerfile 是本次提供的归档构建入口，本次没有执行镜像构建；新构建不保证与历史镜像逐层或 digest 相同。
它保留所选官方镜像的 DeepEP、DeepGEMM、NIXL 和 CUDA 库；不启动服务、不提交集群任务。
观测功能默认关闭，需要按受控运行环境设置 epoch、model、role、完整本地 rank/peer 名单及对应开关；本归档不提供可直接接生产的绑定。

若从源码构建 wheel，请显式设置 `SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0`，以免归档提交历史改变自动生成的版本字符串。

## Git 历史与许可

`upstream-v0.28.0` 标签是原上游 commit 的完整文件树快照；为避免导入无关历史，归档使用独立初始提交，原 commit 身份保存在 manifest 中。
原上游 README 保存为根目录 `README.upstream.md`。上游 LICENSE、NOTICE 和文件版权声明保留。
归档整理、补丁重建和校验由 AI 辅助完成；历史补丁代码按已保存 payload 保留，未借归档之机重写其逻辑。
