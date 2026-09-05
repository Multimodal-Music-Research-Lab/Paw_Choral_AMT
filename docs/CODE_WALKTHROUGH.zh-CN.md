# 代码讲解（中文）

这套代码的主线是：把一段混合合唱音频转换成音符事件，并进一步把每个音符分给女高音、女低音、男高音和男低音（SATB）。

## 三个系统分别做什么

- **PagCT**（`src/models.py::FlexibleHPT`）：只回答“混合音频里有哪些音符”，输出一条合并 MIDI，不区分声部。
- **PawCT**（`src/models.py::FlexibleHPTChoralStream`）：共享一个声学编码器，再用四组声部 head 同时预测 S/A/T/B 的 onset、frame 和 offset；presence head 判断某一段中哪些声部存在；四个声部取 max 得到全局 union。
- **Post-VA**（`src/train_midi_voice_assignment.py`）：先让 PagCT 找音符，再用双向 LSTM 仅根据符号音符序列把音符分给 SATB。它看不到原始声学证据，所以转录错误一旦发生，第二阶段很难补救。

## 数据怎样进入模型

`src/data_generator.py` 先把音频和 MIDI 打包为 HDF5，再按 10 秒窗口采样。`ChoralSATBDataset` 从 `note/<歌曲>.pkl` 读取 S1/S2/A1/A2/T/B 等标注，把同类声部合并为 S/A/T/B，生成 `[时间, 4, 88音高]` 的监督张量。

论文中的三个标注策略也在这里：

1. `part_name`：直接相信原始声部名。
2. `range_prior`（RP）：参考典型 SATB 音域重新分配歧义音符。
3. `ordered_continuity`（OC）：在同一 onset 内保持从高到低的声部顺序，同时惩罚旋律大跳和同一声部重叠。

## PawCT 为什么需要 union loss

`src/losses.py::choral_task_bce` 同时监督四个声部和它们的并集。如果只监督声部，模型可能在困难的声部分配中把本来存在的音符整体丢掉；union loss 强制四个 head 合起来仍覆盖全局音符内容。论文报告中，去掉 union loss 后平均 note F1 为 0.190，加回后为 0.217。

## 推理和评估

`src/inference.py` 把整首音频切窗、批量前向、再拼回完整时间轴。`src/utilities.py` 根据 onset/frame/offset 概率找峰值并生成 note events。`src/calculate_scores.py` 评估全局转录，`src/calculate_choral_scores.py` 逐声部评估 50/100 ms onset F1、含 offset 的 F1、frame F1 和声部 presence。

阈值现在强制先在 validation 上搜索，再冻结后跑 test；validation/test 的 probabilities 也被放进不同目录，避免误覆盖。

## 当前不能过度声称的部分

- 找到的代码与模型、Figure 3 高度对应，但原目录没有纳入 Git，无法用旧 commit 精确锁定论文快照。
- 历史声学 checkpoint 实际跑到 300k，而论文写 200k 和 validation-loss early stopping；当前训练循环没有实现这种 early stopping。
- 当前 loss 没有使用已生成的 onset/offset regression target，因此“由回归监督得到亚帧精度”的文字需要改或补实现。
- PagCT 是三套完整声学分支，PawCT 是共享编码器；二者不是严格等参数对照。
- Post-VA 历史实际设置是 `lr=1e-3`、AdamW、80 epochs，并带 range/crossing 正则，与论文描述需要统一。
- 目前正式 `PianoTranscriber` 只输出合并 union；四轨 SATB 解码逻辑仍在可视化代码里，尚需抽成稳定 CLI。

因此，最准确的定位是：**核心论文实现和作图代码已经找对并整理出来，但论文数值的冻结复现包还没有完成。**
