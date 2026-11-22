# 相位声码器：长音跑调与重叠区起伏

## 题面

把变速不变调换成相位声码器后，长音会逐渐跑调、重叠区还忽大忽小：`istft` 用固定窗和而不是逐样本累计窗和，重叠区幅度随 hop 变化、能量不守恒；`instantaneous_freq` 的相位展开只在 (−π,π) 内取模一次，跨多帧累积后频率整体偏移 2π·k/hop；`phase_advance` 用未展开的相位做积分，长时间合成后相位漂移，`pitch_shift` 输出出现相位断裂；`stretch` 的输出长度按 `int(len/rate)` 取整，rate 为无理数比例时累计误差超过 1 采样；`pitch_shift` 在频域搬移后按原采样率重采样，未补偿 bin 频率偏移，导致音高比例偏差约 1/fft_bin；`stft` 的窗未做周期化修正（用对称窗），帧间重叠求和不是常数，出现周期性 1 个采样的 ripple；`consistency` 用相邻帧相位差的绝对值之和，正负抵消，恒为高值；`roundtrip_error` 只比较前 hop 个样本，重叠区误差被忽略；`istft` 的 `length` 只在末尾截断、不在开头对齐，居中窗情况下整体偏移半个窗；`instantaneous_freq` 对静音帧（幅度 0）仍按相位差分计算，产生高频伪影并把相位推进错误。pvoc/vocoder.py 是核心，tests/test_vocoder.py 15个红。要修的是：8 个桩的实现必须同时满足上述不变量，且只能修改 pvoc/vocoder.py。

这题的坑都在不变量上：窗和守恒、相位展开、bin 频率补偿、往返一致，改一处往往要靠三条测试一起验证。

## 运行方式

在仓库根目录执行（仅需标准库与 pytest，离线即可）：

```text
python -m pytest tests -q
```

初始状态必须恰好是：

```text
15 failed
```

失败来自 `pvoc/vocoder.py` 中 8 个仍为 `raise NotImplementedError` 的桩；不得出现 collection error、import error 或 syntax error。修完后的目标是 `15 passed`，并且不得通过放宽断言、修改测试或引入第三方库来达成。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/test_vocoder.py` 不可修改；它逐条固定了帧长、hop、相位展开、对齐、长度与 `1e-9` 容差等契约。
- 只允许修改 `pvoc/vocoder.py` 中 8 个函数的实现；不得改动公开签名、返回类型或 docstring 中的约定。
- 只使用 `math`、`cmath` 等标准库；测试确定性、无随机、无网络、无 sleep。
- 建议按“窗与重叠相加 → 相位展开 → 合成推进 → 频域搬移”的顺序修，每改一处都回归往返、能量、顺序无关三类测试。
