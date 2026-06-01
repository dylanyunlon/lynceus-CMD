# Lynceus-CMD — Claude 接力开发进度表

> 每位 Claude 在 `lynceus_port/` 上执行"鲁迅拿法"改写：在已有 mv 基础上动态修改
> ~20% 算法内容 + 全链路 `_dbg()` 断点注入，使运行时能像真实开发环境一样得到反馈。

---

## 状态总览

| Claude | 里程碑 | 范围 | 状态 | 交付物 |
|--------|--------|------|------|--------|
| **#1** | **M001–M020** | port层全量改写（34模块） | ✅ **已完成** | `0001-port-lynceus_port-dbg-injection.patch` |
| #2 | M021–M024 | GPU kernel estimation + 分布式 sync 深化 | 🔲 待启动 | — |
| #3 | M025–M028 | Mixed-precision optimizer + Auto-sharding 深化 | 🔲 待启动 | — |
| #4 | M029–M032 | NCCL collector + FSDP compat 深化 | 🔲 待启动 | — |
| #5 | M033–M036 | 端到端实验配置 + 可视化图表生成 | 🔲 待启动 | — |
| #6 | M037–M038 | 集成测试 + 实验面板 + 论文图表 | 🔲 待启动 | — |

---

## Claude #1 完成记录（M001–M020 port层改写）

### 交付统计
- **文件**: 34个 Python 模块 + 1个测试文件
- **改动量**: 35 files changed, ~3092 insertions, ~2051 deletions
- **断点覆盖**: 200+ `_dbg()` 调用覆盖全部关键路径
- **测试**: 13/13 port测试 PASS, 原版 16/16 无回归

### 改写清单

**手工深化改写（4个核心文件）**:

| 文件 | _MOD_TAG | 断点数 | 关键改写 |
|------|----------|--------|----------|
| `schema.py` | SCH | 12 | Welford算法, Kahan求和, sigmoid拥塞模型, Dijkstra visited set |
| `cost_model.py` | CST | 20+ | CPU分离tuple/predicate, GPU分离mem/arith+瓶颈定位, recommend打印runner-up |
| `pipeline_scheduler.py` | PLS | 15+ | 每stage断点, comm_overhead修正, bubble详情 |
| `router.py` | RTR | 10+ | route_one/batch设备分布统计, run_all耗时 |

**批量转换（33个文件）**:
- 变量重命名: `total_cost→aggregate_cost`, `hit_rate→cache_hit_ratio` 等20个映射
- 常量微调: `0.95→0.9475`, `0.5→0.495`, `1e-6→1.05e-6` 等
- videx 7个文件: sub_platforms stub注入 + try/except降级

### 修复的Bug
1. `_dbg` f-string `{{tag}}`→`{tag}` 全量修复（22个文件）
2. 单参数 `_dbg(f"xxx")` → 双参数 `_dbg("tag", f"xxx")`（8个文件）
3. `create_default_topology` 字段名 `random_seek_expense`→`seek_cost`

---

## Claude #2 任务指引（M021–M024）

### M021–M022: GPU kernel cost estimation 深化
**目标文件**: `lynceus_port/gpu_cost_kernel.py` (586行)

当前状态：批量转换已完成（变量重命名+常量微调+_dbg基础注入），但函数级断点太浅（仅6次调用）。

待做：
- [ ] `GPUKernelEstimator.estimate_seq_scan`: 深化内存层级模型断点（L1/L2/HBM分层耗时）
- [ ] `estimate_btree_lookup`: BTree遍历代价每层断点
- [ ] `estimate_hash_probe`: hash碰撞链长度+探测代价断点
- [ ] `estimate_hash_build`: 建表过程bucket分配断点
- [ ] `compare_cpu_vs_gpu`: 对比决策推理链断点
- [ ] `estimate_sort`: radix-sort pass级断点
- [ ] 算法改写 ~20%: 建议修改SM占用率模型或内存带宽计算公式

### M023–M024: 分布式 cost model sync 深化
**目标文件**: `lynceus_port/distributed/sync.py` (行数待查)

待做：
- [ ] push/pull 通信原语每步断点
- [ ] 参数聚合过程断点（梯度平均、权重更新）
- [ ] 网络延迟模拟断点

---

## Claude #3 任务指引（M025–M028）

### M025–M026: Mixed-precision optimizer 深化
**目标文件**: `lynceus_port/distributed/optimizer.py`

待做：
- [ ] 融合优化器步骤断点（loss scale、grad clip、param update）
- [ ] 收敛检测断点深化
- [ ] 算法改写: Adam → 改写内循环变量名 + 等价数学表达

### M027–M028: Auto-sharding 深化
**目标文件**: `lynceus_port/sharding.py`

待做：
- [ ] shard分配决策断点
- [ ] 负载均衡指标断点
- [ ] epoch追踪完整断点链

---

## Claude #4 任务指引（M029–M032）

### M029–M030: NCCL-backend all_reduce collector 深化
**目标文件**: `lynceus_port/distributed/collector.py`

### M031–M032: FSDP compatibility layer 深化
**目标文件**: `lynceus_port/distributed/fsdp_compat.py`

---

## Claude #5 任务指引（M033–M036）

### M033–M034: 端到端实验配置
- [ ] 创建 `configs/` 目录 + 实验配置文件
- [ ] 创建 `scripts/run_benchmark.py` 入口脚本
- [ ] 适配 `run_lynceus.sh` 支持 `lynceus_port`

### M035–M036: 可视化图表生成
- [ ] `viz/plot_panels.py` 深化断点
- [ ] 生成 data_demo 格式的输出数据

---

## Claude #6 任务指引（M037–M038）

### M037: 集成测试
- [ ] 为每对里程碑创建 `tests/test_port_mXXX_mXXX.py`
- [ ] 端到端测试: 拓扑→代价模型→路由→流水线→缓存→FP8→benchmark 全链路

### M038+: 实验面板 + 论文图表
- [ ] TPC-H 负载实验
- [ ] 消融实验面板
- [ ] 论文图表生成

---

## 使用方法

```bash
# 应用 Claude #1 的 patch
cd lynceus-CMD
git am 0001-port-lynceus_port-dbg-injection.patch

# 验证
LYNCEUS_DEBUG=0 python3 -c "import lynceus_port; print('OK')"
python3 tests/test_port_m001_m002.py

# 开启调试运行
LYNCEUS_DEBUG=1 python3 -c "
from lynceus_port.cost_model import *
topo = create_default_topology()
engine = CostModelEngine(topo)
q = QueryDescriptor(query_id='test', query_type=QueryType.FULL_TABLE_SCAN,
                    estimated_rows=1000000, table_rows=1000000)
engine.recommend(q, 'cpu0')
"
```

---

## 通用约定

1. **作者**: `dylanyunlon <dogechat@163.com>`
2. **分支**: `main`
3. **patch格式**: `git format-patch -1 HEAD`
4. **_MOD_TAG**: 每个模块3字母标识，不重复
5. **测试**: 每位Claude完成后必须 `LYNCEUS_DEBUG=0` 全量import + 跑对应测试
6. **调试输出**: 全部到 stderr，stdout 保持干净
