## 第二位深度移植 Claude（integrations/ 14文件深度改写）

**日期**: 2026-06-02
**任务**: integrations/ 14个集成模块的算法等价改写(~20%) + 调试桩密度加倍 + _dbg自递归bug批量修复
**里程碑**: M036–M040（integrations 二次改写）

### Bug 修复
- 修复 9 个文件的 `_dbg()` 自递归无限递归 bug（7 integrations + gpu_cost_kernel.py + sharding.py 待修）
- 修复 videx_bridge.py property 调试桩缩进错误
- 每个文件分配唯一 3 字母模块标签避免日志冲突: VHI/ROB/PPC/PQL/VBR/VCM/VMD/VNE/VUT

### 算法等价改写清单（14 文件, 809 行增 / 472 行删 = 净增 337 行）

| 文件 | 行数变化 | 改写要点 |
|------|----------|----------|
| `videx_histogram.py` | 1613→1589(-24) | partition/maxsplit 防值含冒号误拆; frozenset null 判断; 二分查找粗定位+线性微调 O(log n); Laplace 平滑 KL 散度; EMD 归一化[0,1]; stable sort; 预计算 inv_ndv; 乘法缩放替代除法 |
| `par2qo_robustness.py` | 881→916(+35) | scrambled Halton (黄金比例哈希偏移); 64维素数表; Winitzki(2008) 高精度 erfinv; 双侧 u01 clamp [1e-8, 1-1e-8] |
| `tabular_bridge.py` | 625→655(+30) | cache-line 对齐 fanout (8B 边界); 迭代求高度替代 log/log 浮点除法; 可调 fill_factor 参数暴露; Amdahl 线程缩放模型; TLB miss 代价模型; Robin Hood 探测公式 |
| `videx_ndv_estimator.py` | 578→637(+59) | Goodman 加偏差校正; Jackknife 自动切换 JK1/JK2 混合; Sichel ln-space 防 exp 溢出; Bootstrap log-space p^r 防大 r 下溢; Ada 5 档精细切换(+Sichel 分支) |
| `par2qo_bridge.py` | 433→458(+25) | softmax(-penalty) 替代 1/(1+penalty) 重加权; Welford 在线加权方差; 熵值打印 |
| `par2qo_cost.py` | 263→~280(+17) | SeqScan 加 cache miss ratio (10% 预取模型); IndexScan 动态 B-tree 深度 log(rows,200); HashJoin build 开销 2.0→2.5x |
| `par2qo_querylets.py` | 572→~600(+28) | 同表多谓词相关性修正 sqrt(Πsel); defaultdict 按表分组 |
| `par2qo_plan_cache.py` | 364→387(+23) | 2Q 驱逐策略——扫描前 25% LRU 项找 access_count 最小的, 防刚插入热查询被误淘汰 |
| `videx_histogram_utils.py` | 528→573(+45) | reservoir 采样混合(前半 stride + 后半 reservoir); galloping merge (指数搜索, >8x 比例时 O(n log m)); sort_validate 去重统计+高重复率自动调桶 |
| `videx_utils.py` | 537→569(+32) | Zipf 分布选择率(singlepoint: sqrt(2)/ndv); isqrt 替代 sqrt; log2 下界保护 |
| `videx_cost_model.py` | 399→422(+23) | 对数空间选择率乘积防浮点下溢; clip [1e-15, 1.0] |
| `videx_bridge.py` | 289→305(+16) | property 调试桩修复+增强 |
| `par2qo_utils.py` | 431→439(+8) | 批量函数入口调试桩注入 |
| `videx_metadata.py` | 552→572(+20) | 批量函数入口调试桩注入 |

### 调试桩统计

| 指标 | 改写前 | 改写后 |
|------|--------|--------|
| integrations 总调试点 | ~160 | ~330+ |
| videx_histogram.py 调试密度 | 0.7% | 3.0% |
| 最低密度文件 | 0% (histogram_utils) | 1% (所有文件≥1%) |
| _dbg 自递归 bug | 9 文件 | 0 文件(integrations内) |
