# 代码讲解（中文）

这套代码的主线是：把一段混合合唱音频转换成音符事件，并进一步把每个音符分给女高音、女低音、男高音和男低音（SATB）。

## 三个系统分别做什么

- **PagCT**（`src/models.py::PagCT`）：只回答“混合音频里有哪些音符”，输出一条合并 MIDI，不区分声部。
- **PawCT**（`src/models.py::PawCT`）：共享一个声学编码器，再用四组声部 head 同时预测 S/A/T/B 的 onset、frame 和 offset；presence head 判断某一段中哪些声部存在；四个声部取 max 得到全局 union。
- **Post-VA**（`src/train_midi_voice_assignment.py`）：先让 PagCT 找音符，再用双向 LSTM 仅根据符号音符序列把音符分给 SATB。它看不到原始声学证据，所以转录错误一旦发生，第二阶段很难补救。

## 数据怎样进入模型

`src/data_generator.py` 先把音频和 MIDI 打包为 HDF5，再按 10 秒窗口采样。合唱数据的全局真值不再依赖会覆盖跨声部同音的 merged MIDI：`ChoralUnionDataset` 从完整 `note/<歌曲>.pkl` 构造 PagCT/PawCT 共用的 canonical union；`ChoralSATBDataset` 再把 S1/S2/A1/A2/T/B 等标注合并为 S/A/T/B，生成 `[时间, 4, 88音高]` 的监督张量。只有落在同一模型帧的同音高 attack 会合并，50/100 ms 评测容差不会改变真值；跨窗口长音的 frame/onset mask 也不再被误删。

论文中的三个标注策略也在这里。新版 RP/OC 默认锁定可信的 S/A/T/B、S1/S2 等标签，只推断未知或明确歧义的标签：

1. `part_name`：直接相信原始声部名。
2. `range_prior`（RP）：参考典型 SATB 音域分配歧义音符，也可作为 voice head 的软音域正则。软 RP 使用“越界位置应为负类”的 BCE，因此越自信的错误越会得到强纠正，同时所有可信正标签都从该惩罚中排除。
3. `ordered_continuity`（OC）：使用随时间间隔衰减的旋律连续性和重叠代价，同时保留 divisi。新版软 OC loss 只连接同一声部相邻、单音且无歧义的标注 onset 事件（即使中间有休止），比较“预测音高运动”和“标注音高运动”；同帧多音/divisi 仍由原始 BCE 完整监督，但会切断轨迹链，避免取出一个并不存在的平均音高。它按事件归一化并衰减过远的连接，不会再被大量持续帧稀释，也不会把正确的旋律跳进拉平。`legacy_*` 模式只用于复现旧的全量重标行为。

真实数据审计还发现，现有声学 HDF5 不是 452 首完整 manifest：train
为 376/392，validation 为 28/30，test 为 30/30。因此用这批数据训练时
必须明确写成 `available-audio-434` 协议并冻结 ID hash；RP 应使用实际
376 首 train 的 p01/p99 音域 S 60–79、A 55–74、T 50–70、B 41–62，
而不能混用包含缺失音频标注的统计。

如果设置 `dataset.youchorale_split_dir`，训练、推理、阈值搜索和评分会
统一读取外部冻结的 train/valid/test manifests，而不是继续相信 HDF5 内
旧的 split 属性。代码会检查三份列表两两不重叠、合并后刚好覆盖全部
HDF5 stem，并把与本机路径无关的 manifest identity 写入 checkpoint 和
probability provenance；不同 split 的 checkpoint 不能混用。

## PawCT 为什么需要 union loss

`src/losses.py::choral_task_bce` 同时监督四个声部和它们的并集。如果只监督声部，模型可能在困难的声部分配中把本来存在的音符整体丢掉；union loss 强制四个 head 合起来仍覆盖全局音符内容。论文报告中，去掉 union loss 后平均 note F1 为 0.190，加回后为 0.217。

## 推理和评估

`src/inference.py` 把整首音频切窗、批量前向、再拼回完整时间轴。`src/utilities.py` 根据 onset/frame/offset 概率找峰值并生成 note events。`src/calculate_scores.py` 评估全局转录，`src/calculate_choral_scores.py` 逐声部评估 50/100 ms onset F1、含 offset 的 F1、frame F1 和声部 presence。正式评估的全局与逐声部 reference 都由同一份原始 `note.pkl` 重建，不会再随 RP/OC 训练目标或评分容差改变。

阈值现在强制先在 validation 上搜索，再冻结后跑 test；validation/test 的 probabilities 也被放进不同目录，避免误覆盖。新 checkpoint 保存完整 resolved config、模型/前端身份和随机状态，并默认拒绝任何未解释的 missing/unexpected key 或输入坐标不一致。

## 当前不能过度声称的部分

- 找到的代码与模型、Figure 3 高度对应，但原目录没有纳入 Git，无法用旧 commit 精确锁定论文快照。
- 历史声学 checkpoint 实际跑到 300k，而论文写 200k 和 validation-loss early stopping；新版循环已有显式 validation metric、`best.pth` 和 patience early stopping，但旧表仍须按统一规则重跑。
- 当前 loss 没有使用已生成的 onset/offset regression target，因此“由回归监督得到亚帧精度”的文字需要改或补实现。
- PagCT 是三套完整声学分支，PawCT 是共享编码器；二者不是严格等参数对照。
- Post-VA 历史实际设置是 `lr=1e-3`、AdamW、80 epochs，并带 range/crossing 正则，与论文描述需要统一。
- 目前正式 `ChoralAMTTranscriber` 只输出合并 union；四轨 SATB 解码逻辑仍在可视化代码里，尚需抽成稳定 CLI。

因此，最准确的定位是：**核心论文实现和作图代码已经找对并整理出来，但论文数值的冻结复现包还没有完成。**

另外，`tools/evaluate_label_masking.py` 是一个 CPU-only 的训练标签诊断：
它用固定的 10/25/50% 嵌套遮蔽检查 RP/OC 能否恢复被藏起的声部标签，
并用同一批 note ID 做循环音域负对照。这里得到的是 prior validity，
不是声学转录分数。
