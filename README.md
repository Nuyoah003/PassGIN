## 🧠 项目简介

口令作为最常见的身份验证方式，广泛应用于各类数字系统中。然而，用户为了便于记忆，往往采用“有规则”的方式生成口令，并在多个站点之间进行变体复用。例如，基础口令 `123abc` 可能被用户变换为 `1a2b3c` 以规避检测或提升强度。这类**同构变体**（Isomorphic Variants）极大增加了口令结构的复杂性，对现有的口令猜测模型提出了挑战。

传统模型（如 Markov、PCFG）无法有效建模字符的重新排列模式与邻接性差异，导致对同构结构的识别能力不足。

为此，我们提出了基于图神经网络的口令猜测框架 **PassGIN**：

- 🔗 利用**Graph Isomorphism Network, GIN**对口令进行图建模；
- 🔍 显式建模字符之间的邻接与重排结构，提升对同构口令的判别能力；
- ⚖️ 引入 **PassCluster 动态边权机制**，结合大规模数据集中字符邻接频率增强图结构表达能力。
- 此外，我们扩展实现了两个对比模型：**PassGCN**：基于图卷积网络（GCN）**PassSAGE**：基于 GraphSAGE 模型

---

这里我们展示了同构口令和基础口令的一些示例：

![image](https://github.com/user-attachments/assets/1df18e31-f81f-4645-998e-673b7b95ec69)

这里是PassGIN的图形摘要：

![image](https://github.com/user-attachments/assets/f56ab3c1-f017-4bd2-89dd-86c89de769a7)



## 📂 项目结构与主要代码说明

| 文件 / 文件夹 | 功能说明 |
|---------------|----------|
| `data_handle_4_new.py` | 将原始口令训练集读取并转换为图建模所需的 `.pkl` 格式 |
| `model.py` | 包含图同构网络（GIN）模型定义 |
| `maint.py` | 训练入口脚本，使用 `.pkl` 数据进行训练并输出模型结果 |
| `loadgraph4_new.py` | 构建图数据集并创建 `GraphDataset`（一次性加载） |
| `monte.py` | 基于 Monte Carlo 算法估算测试集中口令的猜测次数（PassGIN） |


---

## ⚙️ 默认训练参数设置（PassGIN）

| 参数 | 默认值 |
|------|--------|
| `--batch-size` | 128 |
| `--lr` | 0.001 |
| `--num_layers` | 5 |
| `--num_mlp_layers` | 2 |
| `--neigh-pooling-type` | "sum" |
| `--num-tasks` | 1 |
| `--hidden-dim` | 64 |
| `--feat-drop` | 0.05 |
| `--final-drop` | 0.05 |

---


## 📌 致谢与说明

感谢使用 PassGIN 项目！我们希望本项目能为口令安全研究者和安全工程实践者提供新的思路，特别是在识别与破解同构口令方面的建模与评估。

📬 若您在使用过程中有任何疑问或建议，欢迎提交 issue 与我们联系。

**一个好消息：我们已经上传了requirements.txt，您可以直接通过pip来解决环境依赖**
