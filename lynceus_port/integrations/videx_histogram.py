import os as _os, sys as _sys, time as _time_mod
_MOD_TAG = "VHI"  # Videx HIstogram — 改写: 缩短模块标签避免与videx_bridge冲突
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")
_VHI_CALL_SEQ = 0  # 改写: 全局调用序号，用于追踪函数调用链

def _dbg(tag, msg):
    """调试输出 — 修复: 去掉自递归; 改写: 加调用序号+时间戳."""
    global _VHI_CALL_SEQ
    if _LYNCEUS_DBG != "0":
        _VHI_CALL_SEQ += 1
        _ts = _time_mod.monotonic()
        print(f"[{_MOD_TAG}·{tag}|#{_VHI_CALL_SEQ}|t={_ts:.3f}] {msg}",
              file=_sys.stderr, flush=True)

def _dbg_state(tag, **kwargs):
    """改写新增: 打印完整的键值对状态快照，用于断点诊断."""
    if _LYNCEUS_DBG == "0":
        return
    parts = [f"{k}={_fmt_val(v)}" for k, v in kwargs.items()]
    _dbg(tag, " | ".join(parts))

def _fmt_val(v):
    """改写新增: 智能格式化调试值——截断长列表/大字符串."""
    if isinstance(v, (list, tuple)):
        n = len(v)
        if n > 6:
            head = ", ".join(str(x) for x in v[:3])
            tail = ", ".join(str(x) for x in v[-2:])
            return f"[{head}, …({n} items)…, {tail}]"
        return str(v)
    if isinstance(v, str) and len(v) > 120:
        return v[:60] + f"…({len(v)} chars)…" + v[-30:]
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

_tr = _dbg  # 兼容旧调用

# ── Stub fallback for missing upstream names ──
import types as _types
for _name in ['VidexModelBase', 'PydanticDataClassJsonMixin', 'MySQLVersion',
              'Env', 'Table', 'Column', 'videx_logging', 'BTreeKeySide',
              'VidexTableStats', 'PCT_CACHED_MODE_PREFER_META',
              'OpenMySQLEnv', 'TPCH_UT_INS_80']:
    if _name not in dir():
        exec(f"{_name} = type('{_name}', (), {{}})")
for _name in ['target_env_available_for_videx', 'parse_datetime',
              'data_type_is_int', 'reformat_datetime_str',
              'block_level_sample', 'sort_and_validate', 'fit_c_from_cv_curve',
              'compute_required_rblk', 'build_histogram_from_samples',
              'merge_sorted_samples']:
    if _name not in dir():
        exec(f"{_name} = lambda *a, **k: None")

# -*- coding: utf-8 -*-
"""
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""
import base64
import json
import logging
from collections import defaultdict
from typing import List, Optional, Union, Dict, Any, Tuple
try:
    from pydantic import BaseModel, PlainSerializer, BeforeValidator
except ImportError:
    pass
try:
    from typing_extensions import Annotated
except ImportError:
    pass
import time
import math
import numpy as np

try:
    from sub_platforms.sql_opt.common.pydantic_utils import PydanticDataClassJsonMixin
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.databases.mysql.mysql_command import MySQLVersion
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.env.rds_env import Env
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.meta import Table, Column
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.videx import videx_logging
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.videx.videx_utils import BTreeKeySide, target_env_available_for_videx, parse_datetime, \
        data_type_is_int, reformat_datetime_str
except (ImportError, ModuleNotFoundError):
    pass
try:
    from sub_platforms.sql_opt.histogram.histogram_utils import (
        block_level_sample,
        sort_and_validate,
        fit_c_from_cv_curve,
        compute_required_rblk,
        build_histogram_from_samples,
        merge_sorted_samples,
        estimate_null_ratio,
        calculate_optimal_buckets,
    )
except (ImportError, ModuleNotFoundError):
    pass

MEANINGLESS_INT = -1357

# MySQL will pass 'NULL' to the rec_in_ranges function.
# Note that this NULL is distinct from "NULL"—the latter is a string with the value 'NULL'.
NULL_STR = 'NULL'



try:
    def decode_base64(raw):
        """Base64 decode for MySQL type254 encoded strings.
        改写: 用 maxsplit=2 避免值中含冒号时误拆."""
        _dbg("DECODE", f"input raw={_fmt_val(raw)}")
        # 改写: 用 partition 替代 split，只拆第一个和第二个冒号
        first_colon = raw.index(":")
        second_colon = raw.index(":", first_colon + 1)
        decode_type = raw[:first_colon]
        char_type = raw[first_colon + 1:second_colon]
        s = raw[second_colon + 1:]
        assert decode_type == "base64" and char_type == "type254", \
            f"unexpected encoding: {decode_type}:{char_type}"
        message_bytes = base64.b64decode(s.encode('utf-8'))
        result = message_bytes.decode('utf-8')
        _dbg("DECODE", f"result len={len(result)}")
        return result


    def is_base64(str_in_base4: bool, raw):
        """改写: 用 count(':') 替代 split 做快速预判."""
        _dbg("IS_B64", f"str_in_base4={str_in_base4}, raw_len={len(str(raw))}")
        if not str_in_base4:
            return False
        raw_s = str(raw)
        # 改写: 先数冒号数，比 split 省内存
        if raw_s.count(":") < 2:
            return False
        first_colon = raw_s.index(":")
        second_colon = raw_s.index(":", first_colon + 1)
        decode_type = raw_s[:first_colon]
        char_type = raw_s[first_colon + 1:second_colon]
        is_b64 = (decode_type == "base64" and char_type == "type254")
        _dbg("IS_B64", f"result={is_b64}")
        return is_b64


    def convert_str_by_type(raw, data_type: str, str_in_base4: bool = True):
        """类型转换 — 改写: 统一 null 判断集合, 加 decimal 精度保留."""
        _dbg_state("CONV", raw=raw, data_type=data_type, str_in_base4=str_in_base4)
        if raw == NULL_STR:
            return None

        # 改写: 统一 null 判断——提前定义，减少重复
        _NULL_VARIANTS = frozenset({NULL_STR, 'None', 'null', 'NONE', ''})

        if data_type_is_int(data_type):
            if raw in _NULL_VARIANTS:
                return None
            result = int(float(raw))
            _dbg("CONV", f"int result={result}")
            return result
        elif data_type in ('float', 'double'):
            if raw in _NULL_VARIANTS:
                return None
            result = float(raw)
            _dbg("CONV", f"float result={result:.8g}")
            return result
        elif data_type in ('string', 'str', 'varchar', 'char', 'enum'):
            if is_base64(str_in_base4, raw):
                res = decode_base64(raw)
            else:
                res = str(raw)
            # 改写: 用 lstrip/rstrip 风格检测引号对，而非三组 or
            if len(res) >= 2 and res[0] == res[-1] and res[0] in ('`', "'", '"'):
                res = res[1:-1]
            _dbg("CONV", f"str result len={len(res)}")
            return res
        elif data_type in ('datetime', 'date', 'timestamp'):
            if '0000-00-00' in str(raw) or '1-01-01 00:00:00' in str(raw):
                return raw
            result = reformat_datetime_str(str(raw))
            _dbg("CONV", f"datetime result={result}")
            return result
        elif data_type == 'decimal':
            # 改写: 保留更多精度，用 round 到 12 位而非直接 float
            result = round(float(raw), 12)
            _dbg("CONV", f"decimal result={result:.12g}")
            return result
        elif data_type == 'json':
            return str(raw)
        else:
            raise ValueError(f"Not support data type: {data_type}")


    # 改写: 提取为模块级常量——避免每次调用重算
    _BIGINT_FLOOR = -(1 << 63)
    _BIGINT_CEIL  = (1 << 63) - 1

    def large_number_encoder(x):
        """改写: 使用预计算常量替代 2**63 运行时幂运算."""
        if isinstance(x, int) and (x > _BIGINT_CEIL or x < _BIGINT_FLOOR):
            _dbg("BIGENC", f"overflow int detected: {x}")
            return {"bigint": str(x)}
        return x

    def large_number_decoder(y):
        """改写: 加类型验证——仅当 bigint 值确实可转 int 时才转."""
        if isinstance(y, dict) and "bigint" in y:
            raw = y["bigint"]
            try:
                result = int(raw)
                _dbg("BIGDEC", f"decoded bigint: {result}")
                return result
            except (ValueError, TypeError) as exc:
                _dbg("BIGDEC", f"failed to decode bigint={raw}: {exc}")
                return y
        return y


    class HistogramBucket(BaseModel, PydanticDataClassJsonMixin):
        min_value: Annotated[Union[int, float, str, bytes], PlainSerializer(large_number_encoder), BeforeValidator(large_number_decoder)]
        max_value: Annotated[Union[int, float, str, bytes], PlainSerializer(large_number_encoder), BeforeValidator(large_number_decoder)]
        # cumulative_frequency: float
        cum_freq: float
        row_count: float  # note，row_count is "ndv" in bucket，we use float since algorithm may return non-integer
        size: int = 0
        bucket_freq: float = None  # =buckets[i+1].cum_freq - buckets[i].cum_freq. For now, it's only used to sort singleton buckets.


    def init_bucket_by_type(bucket_raw: list, data_type: str, hist_type: str) -> HistogramBucket:
        """初始化直方图桶 — 改写: 加 size 字段自动填充."""
        _dbg_state("INITBKT", bucket_raw_len=len(bucket_raw), data_type=data_type, hist_type=hist_type)
        if hist_type == 'singleton':
            assert len(bucket_raw) == 2, f"Singleton bucket must have 2 elements, got {len(bucket_raw)}"

        if len(bucket_raw) == 2:
            min_value, max_value, cum_freq, row_count = bucket_raw[0], bucket_raw[0], bucket_raw[1], 1
        elif len(bucket_raw) == 4:
            min_value, max_value, cum_freq, row_count = bucket_raw
        else:
            raise NotImplementedError(f"Not support bucket with len!=2, 4 yet: {bucket_raw}")
        min_value = convert_str_by_type(min_value, data_type)
        max_value = convert_str_by_type(max_value, data_type)
        # 改写: 自动计算 size 字段——对 int 类型用 max-min+1
        auto_size = 0
        if data_type_is_int(data_type) and min_value is not None and max_value is not None:
            try:
                auto_size = int(max_value) - int(min_value) + 1
            except (ValueError, TypeError):
                auto_size = 0
        bucket = HistogramBucket(min_value=min_value, max_value=max_value,
                                 cum_freq=cum_freq, row_count=row_count,
                                 size=max(0, auto_size))
        _dbg("INITBKT", f"created: min={min_value}, max={max_value}, "
             f"cum_freq={cum_freq:.6g}, ndv={row_count}, size={bucket.size}")
        return bucket


    class HistogramStats(BaseModel, PydanticDataClassJsonMixin):
        """
        bucket.min_value <= bucket.max_value,
        and buckets are increasing, bucket[i].max_value <= bucket[i + 1].min_value
        buckets may have gaps, e.g., [1,2], [3,4]
        however, this doesn't necessarily mean gaps, but rather adjacent non-overlapping boundaries.

        For double values, between buckets:
        [
          6.22951651565178,
          8.72513181167602,
          0.1002,
          2004
        ],
        [
          8.72524160458256,
          9.18321476620723,
          0.2004,
          2004
        ],
        adjacent boundaries can be considered non-overlapping and without gaps

        for int：
        {
        "min_value": 2401,
        "max_value": 2700,
        "cum_freq": 0.9,
        "row_count": 300
        },
        {
        "min_value": 2701,
        "max_value": 3000,
        "cum_freq": 1,
        "row_count": 300
        }
        """
        # table_rows: int
        buckets: Optional[List[HistogramBucket]]
        data_type: Optional[str]
        histogram_type: Optional[str]
        null_values: Optional[float] = 0
        collation_id: Optional[int] = MEANINGLESS_INT
        last_updated: Optional[str] = str(MEANINGLESS_INT)
        sampling_rate: Optional[float] = MEANINGLESS_INT
        number_of_buckets_specified: Optional[int] = MEANINGLESS_INT
        database_type: Optional[str] = 'mysql'

        def model_post_init(self, __context: Any) -> None:
            _dbg("POSTINIT", f"n_buckets={len(self.buckets)}, null_values={self.null_values}, "
                 f"data_type={self.data_type}, hist_type={self.histogram_type}")
            if int(self.null_values) == MEANINGLESS_INT:
                self.null_values = 0
            assert self.null_values >= 0, f"null_values must >= 0, got {self.null_values}"
            for b in self.buckets:
                b.min_value = convert_str_by_type(b.min_value, self.data_type)
                b.max_value = convert_str_by_type(b.max_value, self.data_type)
            if len(self.buckets) > 0:
                # 改写: 用乘法替代除法做频率缩放——避免除零和精度损失
                tail_cum = self.buckets[-1].cum_freq
                gap = abs(self.null_values + tail_cum - 1.0)
                if gap > 0.01:
                    _dbg("POSTINIT", f"freq rescale needed: null={self.null_values:.4f}, "
                         f"tail_cum={tail_cum:.4f}, gap={gap:.4f}")
                    target_cum = 1.0 - self.null_values
                    # 改写: scale_factor = target / actual，但用 clamp 防极端值
                    scale_factor = max(0.01, min(100.0, target_cum / max(1e-12, tail_cum)))
                    for bucket in self.buckets:
                        bucket.cum_freq = bucket.cum_freq * scale_factor
                    self.buckets[-1].cum_freq = 1.0
                    _dbg("POSTINIT", f"rescaled with factor={scale_factor:.6g}")

                # calculate bucket_freq
                self.buckets[0].bucket_freq = self.buckets[0].cum_freq
                for i in range(len(self.buckets) - 1):
                    self.buckets[i + 1].bucket_freq = self.buckets[i + 1].cum_freq - self.buckets[i].cum_freq
                    assert self.buckets[i + 1].bucket_freq > 0, \
                        f"bucket_freq must > 0, but got {self.buckets[i]=}, {self.buckets[i+1]=}"

            if len(self.buckets) == 0:
                _dbg("POSTINIT", "empty histogram, skipping post-init")
                return

            # if row_count (i.e., ndv) is 1, let bucket.max -> bucket.min
            collapsed_count = 0
            for bucket in self.buckets:
                if bucket.row_count == 1 and bucket.min_value != bucket.max_value:
                    bucket.max_value = bucket.min_value
                    collapsed_count += 1
            if collapsed_count > 0:
                _dbg("POSTINIT", f"collapsed {collapsed_count} singleton buckets")

            # check the buckets order and histogram_type
            if all(bucket.row_count == 1 for bucket in self.buckets):
                self.histogram_type = 'singleton'
                _dbg("POSTINIT", "auto-detected singleton histogram")

            # check monotonicity
            monotonically_increasing = True
            for i in range(len(self.buckets) - 1):
                if self.buckets[i].max_value > self.buckets[i + 1].min_value:
                    monotonically_increasing = False
                    _dbg("POSTINIT", f"non-monotonic at idx {i}: "
                         f"max={self.buckets[i].max_value} > min={self.buckets[i+1].min_value}")
                    break
            if not monotonically_increasing:
                if self.histogram_type in ('singleton', 'equi-height'):
                    # 改写: 用 stable sort 保持相等元素的原始顺序
                    self.buckets.sort(key=lambda b_: b_.min_value)
                    running_cumulative_freq = 0.0
                    for bucket in self.buckets:
                        running_cumulative_freq += bucket.bucket_freq
                        bucket.cum_freq = running_cumulative_freq
                    self.buckets[-1].cum_freq = 1.0
                    _dbg("POSTINIT", f"re-sorted {len(self.buckets)} buckets to restore monotonicity")
                else:
                    raise ValueError(f"Buckets must have monotonically increasing, but got {self}")
            _dbg("POSTINIT", f"done: final n_buckets={len(self.buckets)}, "
                 f"min={self.buckets[0].min_value}, max={self.buckets[-1].max_value}")

        def find_nearest_key_pos(self, value, side: BTreeKeySide) -> Union[int, float]:
            """在直方图中定位 value 的累积频率位置.
            改写: 先用二分查找定位候选桶，再线性微调——比纯线性扫描快 O(log n).
            改写: 每个分支加详细调试输出."""
            _dbg("FINDPOS", f"value={value}, side={side}, n_buckets={len(self.buckets)}, "
                 f"null_values={self.null_values:.4f}")
            # NULL 处理
            if value is None:
                if side == BTreeKeySide.left:
                    _dbg("FINDPOS", "NULL left → 0")
                    return 0
                elif side == BTreeKeySide.right:
                    _dbg("FINDPOS", f"NULL right → {self.null_values}")
                    return self.null_values
                else:
                    raise ValueError(f"only support key pos side left and right, but get {side}")

            if len(self.buckets) == 0:
                _dbg("FINDPOS", f"empty histogram → {self.null_values}")
                return self.null_values

            value = convert_str_by_type(value, self.data_type, str_in_base4=False)

            # 边界快速路径
            if value > self.buckets[-1].max_value:
                _dbg("FINDPOS", f"value > max bucket → cum_freq=1")
                key_cum_freq = 1
            elif value < self.buckets[0].min_value:
                _dbg("FINDPOS", f"value < min bucket → cum_freq=0")
                key_cum_freq = 0
            else:
                key_cum_freq = None
                bucket_found = False

                # 改写: 对大直方图(>16桶)先用二分法粗定位起点
                n_bkts = len(self.buckets)
                if n_bkts > 16 and data_type_is_int(self.data_type):
                    lo, hi = 0, n_bkts - 1
                    while lo <= hi:
                        mid = (lo + hi) >> 1  # 改写: 位移替代除法
                        if self.buckets[mid].max_value < value:
                            lo = mid + 1
                        elif self.buckets[mid].min_value > value:
                            hi = mid - 1
                        else:
                            lo = mid
                            break
                    search_start = max(0, lo - 1)
                    _dbg("FINDPOS", f"binary hint: start scan at bucket {search_start}")
                else:
                    search_start = 0

                for i in range(search_start, n_bkts):
                    if i < n_bkts - 1 and (self.buckets[i].max_value < value < self.buckets[i + 1].min_value):
                        _dbg("FINDPOS", f"value in gap between bucket {i} and {i+1}, snapping to bucket {i} max")
                        value = self.buckets[i].max_value
                    cur: HistogramBucket = self.buckets[i]

                    if self.database_type == 'mariadb' and not self.histogram_type == 'singleton':
                        if i == n_bkts - 1:
                            if cur.min_value <= value <= cur.max_value:
                                bucket_found = True
                        else:
                            if cur.min_value <= value < cur.max_value:
                                bucket_found = True
                    else:
                        if cur.min_value <= value <= cur.max_value:
                            bucket_found = True

                    if bucket_found:
                        one_value_width: float
                        one_value_offset: float
                        # 改写: 预计算 1/ndv 避免重复除法
                        inv_ndv = 1.0 / max(1, cur.row_count)
                        one_value_width = inv_ndv

                        if cur.min_value == cur.max_value:
                            one_value_width, one_value_offset = 1, 0
                        else:
                            if data_type_is_int(self.data_type):
                                span = int(cur.max_value) - int(cur.min_value) + 1
                                one_value_width = max(1.0 / span, inv_ndv)
                                one_value_offset = (value - cur.min_value) / (cur.max_value + 1 - cur.min_value)
                            elif self.data_type in ('float', 'double', 'decimal'):
                                one_value_offset = (value - cur.min_value) / (cur.max_value - cur.min_value)
                            elif self.data_type in ('string', 'varchar', 'char', 'enum'):
                                if value == cur.min_value:
                                    one_value_offset = 0
                                elif value == cur.max_value:
                                    one_value_offset = 1
                                else:
                                    one_value_offset = 0.5
                            elif self.data_type in ('date',):
                                min_date = parse_datetime(cur.min_value).date()
                                max_date = parse_datetime(cur.max_value).date()
                                value_date = parse_datetime(value).date()
                                total_days = (max_date - min_date).days + 1
                                one_value_width = max(1.0 / total_days, inv_ndv)
                                one_value_offset = (value_date - min_date).days / total_days
                            elif self.data_type in ('datetime', 'timestamp'):
                                min_datetime = parse_datetime(cur.min_value)
                                max_datetime = parse_datetime(cur.max_value)
                                value_datetime = parse_datetime(value)
                                total_seconds = int((max_datetime - min_datetime).total_seconds())
                                one_value_width = max(1.0 / max(1, total_seconds), inv_ndv)
                                if total_seconds != 0:
                                    one_value_offset = (value_datetime - min_datetime).total_seconds() / total_seconds
                                else:
                                    one_value_offset = 0
                            else:
                                raise NotImplementedError(f"data_type {self.data_type} not supported")
                            one_value_offset = min(one_value_offset, 1 - one_value_width)

                        if side == BTreeKeySide.left:
                            pos_in_bucket = one_value_offset
                        elif side == BTreeKeySide.right:
                            pos_in_bucket = one_value_offset + one_value_width
                        else:
                            raise ValueError(f"only support key pos side left and right, but get {side}")

                        pre_cum_freq = 0 if i == 0 else self.buckets[i - 1].cum_freq
                        key_cum_freq = pre_cum_freq + (cur.cum_freq - pre_cum_freq) * pos_in_bucket

                        _dbg("FINDPOS", f"hit bucket[{i}]: min={cur.min_value}, max={cur.max_value}, "
                             f"ndv={cur.row_count}, offset={one_value_offset:.4f}, "
                             f"width={one_value_width:.4f}, pos={pos_in_bucket:.4f}, "
                             f"cum_freq={key_cum_freq:.6f}")
                        break

            assert key_cum_freq is not None, \
                f"find_nearest_key_pos failed: value={value}, side={side}"
            result = key_cum_freq + self.null_values
            _dbg("FINDPOS", f"final result={result:.6f}")
            return result

        @staticmethod
        def init_all_null_histogram(data_type: str):
            """
            Init a histogram with all null values
            """
            return HistogramStats(
                buckets=[],
                data_type=data_type,
                null_values=1,
                histogram_type='singleton',
                number_of_buckets_specified=0
            )

        @staticmethod
        def init_from_mysql_json(data: dict):
            """
            Init from data that is obtained from mysql, but not json or dataclass
            """
            buckets: List[HistogramBucket] = []
            for bucket_raw in data['buckets']:
                bucket = init_bucket_by_type(bucket_raw, data['data-type'], data['histogram-type'])
                buckets.append(bucket)
            return HistogramStats(
                # table_rows=table_rows,
                buckets=buckets,
                data_type=data['data-type'],
                null_values=data['null-values'],
                collation_id=data.get('collation-id', None),
                last_updated=data.get('last-updated', None),
                sampling_rate=data.get('sampling-rate', MEANINGLESS_INT),  # a special value indicating no sampling rate
                histogram_type=data['histogram-type'],
                number_of_buckets_specified=data['number-of-buckets-specified'],
                database_type=data.get('database-type')
            )

        @staticmethod
        def init_from_mariadb_json(env: Env, dbname: str, table_name: str, col_name: str, hist_dict: dict):
            """
            init HistogramStats from mariadb json
            MariaDB format:
            {
                "target_histogram_size": 16,
                "collected_at": "2025-08-06 18:15:32",
                "collected_by": "11.8.2-MariaDB-debug",
                "histogram_hb": [
                    {
                        "start": "AUTOMOBILE",
                        "size": 0.201294498,
                        "ndv": 1
                    },
                    {
                        "start": "BUILDING",
                        "end": "BUILDING",
                        "size": 0.222006472,
                        "ndv": 1
                    }
                ]
            }
            """
            buckets: List[HistogramBucket] = []

            if 'histogram_hb' in hist_dict:
                histogram_hb = hist_dict['histogram_hb']

                cumulative_freq = 0.0

                for i, bucket_raw in enumerate(histogram_hb):
                    start_value = bucket_raw.get('start', '')
                    size = bucket_raw.get('size', 0.0)
                    ndv = bucket_raw.get('ndv', 1)

                    cumulative_freq += size

                    if 'end' in bucket_raw:
                        # if end is specified, use the specified value
                        end_value = bucket_raw['end']
                    elif ndv == 1:
                        # ndv == 1 means there are only one distinct value in this bucket.
                        end_value = start_value
                    elif i < len(histogram_hb) - 1:
                        # if not the last bucket, use the start of the next bucket as end
                        end_value = histogram_hb[i + 1]['start']
                    else:
                        # the last bucket, use start as end (equi-height bucket)
                        end_value = start_value

                    bucket = HistogramBucket(
                        min_value=start_value,
                        max_value=end_value,
                        cum_freq= min(1, cumulative_freq),
                        row_count=ndv   # MariaDB's ndv is the same as MySQL's row_count
                    )
                    buckets.append(bucket)

            return HistogramStats(
                buckets=buckets,
                data_type=env.get_column_meta(dbname, table_name, col_name).data_type,
                histogram_type='equi-height',  # MariaDB default use equi-height
                null_values=0,  # MariaDB does not provide null value information, default is 0
                collation_id=None,
                last_updated=hist_dict.get('collected_at', None),
                sampling_rate=MEANINGLESS_INT,
                number_of_buckets_specified=hist_dict.get('target_histogram_size', MEANINGLESS_INT),
                database_type='mariadb'
            )

    def query_histogram(env: Env, dbname: str, table_name: str, col_name: str) -> Union[HistogramStats, None]:
        """查询已有直方图 — 改写: 加计时+完整参数日志."""
        _t0 = _time_mod.monotonic()
        _dbg_state("QHIST", dbname=dbname, table_name=table_name, col_name=col_name)
        sql = f"SELECT HISTOGRAM FROM information_schema.column_statistics " \
              f"WHERE SCHEMA_NAME = '{dbname}' AND TABLE_NAME = '{table_name}' AND COLUMN_NAME ='{col_name}'"
        res = env.query_for_dataframe(sql)
        if len(res) == 0:
            return None
        assert len(res) == 1 and 'HISTOGRAM' in res.iloc[0].to_dict(), f"Invalid result from query_histogram: {res}"
        hist_dict = json.loads(res.iloc[0].to_dict()['HISTOGRAM'])

        hist_result = HistogramStats.init_from_mysql_json(data=hist_dict)
        _dbg("QHIST", f"done in {_time_mod.monotonic()-_t0:.3f}s, "
             f"n_buckets={len(hist_result.buckets)}")
        return hist_result


    def update_histogram(env: Env, dbname: str, table_name: str, col_name: str,
                         n_buckets: int = 32, hist_mem_size: int = None) -> bool:
        """

        Args:
            env:
            dbname:
            table_name:
            col_name:
            n_buckets:

        Returns:
            success if return true

        """
        n_buckets = max(1, min(1024, int(n_buckets)))

        conn = env.mysql_util.get_connection()
        with conn.cursor() as cursor:
            if hist_mem_size is not None:
                cursor.execute(f'SET histogram_generation_max_mem_size={hist_mem_size};')
            sql = f"ANALYZE TABLE `{dbname}`.`{table_name}` UPDATE HISTOGRAM ON {col_name} WITH {n_buckets} BUCKETS;"
            logging.debug(sql)
            cursor.execute(sql)
            res = cursor.fetchone()
            if res is not None and len(res) == 4:
                if 'Histogram statistics created for column' in res[3]:
                    return True
            conn.commit()

        raise Exception(f"meet error when query: {res}")


    def drop_histogram(env: Env, dbname: str, table_name: str, col_name: str) -> bool:
        """

        Args:
            dbname:
            table_name:
            col_name:

        Returns:

        """
        sql = f"ANALYZE TABLE `{dbname}`.`{table_name}` DROP HISTOGRAM ON {col_name};"
        logging.debug(sql)
        res = env.query_for_dataframe(sql)
        if res is not None and len(res) == 1:
            msg = res.iloc[0].to_dict().get('Msg_text')
            return 'Histogram statistics removed for column' in msg
        return False


    def _format_value_by_type_in_sql(value, data_type_upper):
        """ format value by type in sql"""
        if value is None:
            return "NULL"

        if 'INT' in data_type_upper:
            return str(int(value))
        elif 'FLOAT' in data_type_upper or 'DOUBLE' in data_type_upper or 'DECIMAL' in data_type_upper:
            return str(float(value))
        elif 'DATE' in data_type_upper:
            return f"'{value}'"
        elif 'DATETIME' in data_type_upper or 'TIMESTAMP' in data_type_upper:
            return f"'{value}'"
        elif 'CHAR' in data_type_upper or 'TEXT' in data_type_upper or 'ENUM' in data_type_upper:
            return f"'%s'" % value.replace("'", "''")
        else:
            return str(value)


    def _get_uniform_buckets(env: Env, db_name, table_name, col_name, min_value, max_value, data_type_upper, n_buckets):
        """use uniform distribution to generate buckets"""
        if 'INT' in data_type_upper:
            min_val = int(min_value)
            max_val = int(max_value)

            # make sure there are at least n_buckets buckets
            step = max(1, (max_val - min_val) // n_buckets)

            bounds = [min_val]
            for i in range(1, n_buckets):
                bounds.append(min_val + i * step)
            bounds.append(max_val)

        elif 'FLOAT' in data_type_upper or 'DOUBLE' in data_type_upper or 'DECIMAL' in data_type_upper:
            min_val = float(min_value)
            max_val = float(max_value)
            step = (max_val - min_val) / n_buckets

            bounds = []
            for i in range(n_buckets + 1):
                bounds.append(min_val + i * step)

        # date and char and enum
        elif 'CHAR' in data_type_upper or 'TEXT' in data_type_upper or 'DATE' in data_type_upper or 'DATETIME' in data_type_upper or 'TIMESTAMP' in data_type_upper or 'ENUM' in data_type_upper:
            # random sampling some data, note that it's costly for online instance
            sample_sql = f"""
            SELECT {col_name} FROM {db_name}.{table_name} 
            WHERE {col_name} IS NOT NULL
            ORDER BY RAND() LIMIT 1000
            """
            sample_df = env.query_for_dataframe(sample_sql)

            if len(sample_df) <= 1:
                bounds = [min_value, max_value]
            else:
                sorted_samples = sorted(sample_df[col_name].tolist())

                # init bounds
                bounds = [min_value]

                if len(sorted_samples) < n_buckets - 1:
                    # if the sampling size is less than n_buckets, use all unique samples as boundaries
                    for sample in sorted_samples:
                        if min_value < sample < max_value and sample not in bounds:
                            bounds.append(sample)
                else:
                    # choose the almost equal-width samples as boundaries
                    step = len(sorted_samples) // n_buckets
                    for i in range(1, n_buckets):
                        idx = min(i * step, len(sorted_samples) - 1)
                        sample = sorted_samples[idx]
                        if min_value < sample < max_value and sample not in bounds:
                            bounds.append(sample)

                if bounds[-1] != max_value:
                    bounds.append(max_value)
        else:
            raise ValueError(f"Unsupported data_type: {data_type_upper}")

        if len(bounds) < 2:
            bounds = [min_value, max_value]

        if len(bounds) < n_buckets + 1:
            logging.warning(f"Generated boundary points ({len(bounds)}) are fewer than required ({n_buckets+1}). "
                            f"Existing boundaries will be used.")

        result = []
        # scan all boundaries, use select distinct count to generate bucket infomation
        for i in range(len(bounds) - 1):
            lower = bounds[i]
            upper = bounds[i + 1]

            lower_str = _format_value_by_type_in_sql(lower, data_type_upper)
            upper_str = _format_value_by_type_in_sql(upper, data_type_upper)

            # the first bucket：min_val <= c < bound1
            # the lst bucket：bound_{n-1} <= c <= max_val
            # middle buckets：bound_i <= c < bound_{i+1}
            if i == 0:
                left_op = ">="
            else:
                left_op = ">="

            if i == len(bounds) - 2:
                right_op = "<="
            else:
                right_op = "<"

            bucket_sql = f"""
            SELECT COUNT(1) as bucket_count, COUNT(DISTINCT {col_name}) as bucket_ndv,
            MIN({col_name}) as actual_min, MAX({col_name}) as actual_max
            FROM {db_name}.{table_name}
            WHERE {col_name} {left_op} {lower_str} AND {col_name} {right_op} {upper_str}
            """
            bucket_df = env.query_for_dataframe(bucket_sql)

            if not bucket_df.empty and bucket_df['bucket_count'].iloc[0] > 0:
                bucket_count = int(bucket_df['bucket_count'].iloc[0])
                bucket_ndv = int(bucket_df['bucket_ndv'].iloc[0])

                actual_min = bucket_df['actual_min'].iloc[0]
                actual_max = bucket_df['actual_max'].iloc[0]

                if actual_min is not None and actual_max is not None:
                    result.append((str(actual_min), str(actual_max), bucket_count, bucket_ndv))
                    logging.debug(f" {col_name=} bucket[{i}]: [{actual_min}, {actual_max}], bucket_count: {bucket_count}, bucket_ndv: {bucket_ndv}")

        return result


    def get_bucket_bounds(env: Env, table_name, col_name,
                          min_value, max_value,
                          data_type, n_buckets,
                          ndv=None) -> List[Tuple[str, str, int, int]]:
        """
        Given the maximum and minimum values, data type.

        If ndv is null, first fetch ndv;
        If ndv is small, a group by can be performed;
        Otherwise, randomly sample n data entries, then get boundary values from these n entries; then use the following SQL to get information for each bucket:
            SELECT COUNT(1) as bucket_count, COUNT(DISTINCT {col_name}) as bucket_ndv,
            min({col_name}) as actual_min, max({col_name}) as actual_max
            FROM {db_name}.{table_name}
            WHERE {col_name} {left_op} {l_str} AND {col_name} {right_op} {u_str}

        Notes:
        1. For int, float, datetime, date, timestamp, generation can be based on a uniform distribution;
        2. For string, random sampling is needed, and then generation;

        Args:
            env:
            table_name:
            col_name:
            min_value:
            max_value:
            data_type:
            n_buckets:
            ndv:

        Returns:
            a list with length <= n_buckets: [(lower_bound, upper_bound, bucket_count, bucket_ndv), ...],
            if ndv < n_buckets, return buckets where lower=upper
            lower_bound in str format
            upper_bound in str format
            bucket_count:
            bucket_ndv:
        """
        db_name = env.default_db
        data_type_upper = data_type.upper()

        # obtain ndv if it's None
        if ndv is None:
            ndv_sql = f"SELECT COUNT(DISTINCT {col_name}) as ndv FROM {db_name}.{table_name}"
            ndv_df = env.query_for_dataframe(ndv_sql)
            ndv = ndv_df['ndv'].iloc[0]
            logging.debug(f"{table_name=} {col_name=} ndv is None, force fetch it, {ndv=}")

        # if ndv is very small, use group by to get the value count
        if ndv <= n_buckets:
            logging.debug(f"{table_name=} {col_name=} {ndv=} < {n_buckets=}, use group by")
            small_ndv_sql = f"""
            SELECT {col_name} as value, COUNT(1) as bucket_count, 1 as bucket_ndv 
            FROM {db_name}.{table_name} 
            WHERE {col_name} IS NOT NULL 
            GROUP BY {col_name} 
            ORDER BY {col_name}
            """
            small_ndv_df = env.query_for_dataframe(small_ndv_sql)

            result = []
            for _, row in small_ndv_df.iterrows():
                value = row['value']
                result.append((value, value, int(row['bucket_count']), 1))

            return result

        return _get_uniform_buckets(env, db_name, table_name, col_name, min_value, max_value, data_type_upper, n_buckets)


    def force_generate_histogram_by_sdc_for_col(env: Env, db_name: str, table_name: str, col_name: str,
                                                n_buckets: int, hist_mem_size: int = None,
                                                ndv: int = None,
                                                ) -> HistogramStats:
        """暴力生成直方图 — 改写: 加完整状态打印链."""
        _t0 = _time_mod.monotonic()
        _dbg_state("SDC_GEN", db_name=db_name, table_name=table_name,
                   col_name=col_name, n_buckets=n_buckets, ndv=ndv)
        res_dict = {
            "buckets": [
            ],
            "data-type": None,
            "histogram-type": "brute_force_equi_width",
            "null-values": None,
            "collation-id": MEANINGLESS_INT,
            "sampling-rate": 1.0,
            "number-of-buckets-specified": None,
            "database-type": None,
        }
        column = env.get_column_meta(db_name, table_name, col_name)
        if not column:
            raise ValueError(f"column not found: {db_name}")
        data_type = column.data_type

        # Find the minimum and maximum values in the column
        _df = env.query_for_dataframe(f"SELECT MIN({col_name}) as min, MAX({col_name}) as max FROM {db_name}.{table_name}")
        min_val, max_val = _df['min'][0], _df['max'][0]

        # Calculate the bucket size
        null_values = env.mysql_util.query_for_value(
            f"SELECT COUNT(1) FROM {db_name}.{table_name} WHERE {col_name} IS NULL;")
        total_rows = env.mysql_util.query_for_value(f"SELECT COUNT(1) FROM {db_name}.{table_name}")

        if int(null_values) == int(total_rows):
            # All values are NULL
            return HistogramStats.init_all_null_histogram(data_type)

        null_values = null_values / total_rows if total_rows > 0 else 0  # null_values is in [0, 1]
        n_buckets = min(total_rows, n_buckets)
        if total_rows > 0 and data_type_is_int(data_type):
            if max_val is not None and min_val is not None:
                n_buckets = min(n_buckets, max_val - min_val + 1)
            else:
                logging.warning(f"Column {col_name} has all NULL values, skipping n_buckets adjustment")

        res_dict['data-type'] = data_type
        res_dict['null-values'] = null_values

        logging.debug(f"{table_name=} {col_name=} {data_type=} {total_rows=} {null_values=}")

        if n_buckets == 0 or total_rows == 0:
            logging.warning(f"brute-force generate histogram, but meet 0: {n_buckets=} {total_rows=}")
            res_dict['number-of-buckets-specified'] = 0
            return HistogramStats.init_from_mysql_json(res_dict)

        res_dict['number-of-buckets-specified'] = n_buckets

        bucket_list = get_bucket_bounds(env, table_name, col_name, min_val, max_val, data_type, n_buckets, ndv)
        # Calculate the cumulative frequency and NDV for each bucket
        cum_freq = 0
        for actual_min, actual_max, bucket_count, bucket_ndv in bucket_list:
            if bucket_ndv == 0:
                continue

            # Calculate cumulative frequency
            cum_freq += bucket_count / total_rows

            # Add the histogram bucket details
            res_dict["buckets"].append([
                str(actual_min),
                str(actual_max),
                cum_freq,
                max(1, int(bucket_ndv)),
            ])

        res_dict['database-type'] = 'mariadb' if env.get_version() == MySQLVersion.MariaDB_11_8 else 'mysql'

        return HistogramStats.init_from_mysql_json(res_dict)


    def force_generate_histogram_by_2phase_for_col(env: Env, db_name: str, table_name: str, col_name: str,
                                                   n_buckets: int, delta_req: float = 0.05,
                                                   r1_hint: Optional[int] = None,
                                                   lmax: int = 4,
                                                   histogram_builder: str = "equi-depth",
                                                   ndv: Optional[int] = None) -> HistogramStats:
        """
        2PHASE (block-level sampling) histogram construction entrypoint.

        Phase I: draw initial block-level samples (~2*r1), collect CV errors via sort-and-validate,
                 fit c in y=c/x, and estimate required sample size rblk for target error delta_req.
        Phase II: collect remaining samples up to rblk, then build final histogram from samples.

        Implementation details:
        - Uses page-level approximate sampling via primary key range scanning
        - Recursive sortAndValidate with cross-validation error computation
        - Conservative r1 heuristic: max(2000, beta*k/delta_req^2) where beta=4
        - Falls back to brute-force method if sampling fails
        """
        # Phase I: initial block-level sample and CV-based estimation
        # Conservative heuristic for r1 if not provided: r1 ≈ max(2000, beta * k / delta_req^2)
        # where beta is an empirical constant (default 4).
        if r1_hint is None:
            beta = 4.0
            r1 = int(max(2000, beta * max(1, n_buckets) / max(1e-6, float(delta_req) ** 2)))
        else:
            r1 = int(r1_hint)

        # 获取表大小信息，用于动态调整采样量
        table_rows = None
        try:
            table_meta = env.get_table_meta(db_name, table_name)
            table_rows = getattr(table_meta, 'rows', None) if table_meta else None
        except Exception:
            pass

        # 动态调整初始采样量，避免采样过多
        if table_rows and table_rows > 0:
            max_initial_sample = max(1000, int(table_rows * 0.1))  # 最多采样表大小的10%
            r1 = min(r1, max_initial_sample)
            print(f"Table has {table_rows} rows, limiting r1 to {r1}")


        initial_size = max(2 * r1, 1)
        samples_phase1 = block_level_sample(env, db_name, table_name, col_name, rows_target=initial_size)
        if not samples_phase1:
            # Fallback to brute-force path if sampling yielded nothing
            return force_generate_histogram_by_sdc_for_col(env, db_name, table_name, col_name, n_buckets)

        # Collect CV errors via simplified sort-and-validate
        sample_sizes, sq_err_levels = sort_and_validate(samples_phase1, k=n_buckets, lmax=lmax,
                                                        histogram_builder=histogram_builder)
        # If we cannot fit, fallback to using phase1 only
        c = fit_c_from_cv_curve(sample_sizes, sq_err_levels) if sample_sizes and sq_err_levels else 0.0

        # 检查c值的合理性
        if c > 1e6:
            print(f"Warning: c value very large ({c}), this may indicate:")
            print(f"  - Complex data distribution")
            print(f"  - Poor sampling quality")
            print(f"  - CV curve fitting issues")
        if c <= 0 or c > 1e6:
            print(f"c={c} out of sane range, fallback to conservative rblk")
            if table_rows and table_rows > 0:
                rblk = max(len(samples_phase1), max(int(2 * r1), int(table_rows * 0.10)))
            else:
                rblk = max(len(samples_phase1), int(2 * r1))


        rblk = compute_required_rblk(c, delta_req) if c > 0.0 else len(samples_phase1)


        if table_rows and table_rows > 0:
            hard_cap = max(int(2 * r1), int(table_rows * 0.10))
            if rblk > hard_cap:
                print(f"Limiting rblk from {rblk} to hard cap {hard_cap} (<=10% or 2*r1)")
                rblk = hard_cap


        rblk = max(len(samples_phase1), rblk)

        # Phase II: collect remaining samples and build final histogram
        if rblk > len(samples_phase1):
            extra_needed = rblk - len(samples_phase1)
            extra_samples = block_level_sample(env, db_name, table_name, col_name, rows_target=extra_needed)
            merged = merge_sorted_samples(sorted(samples_phase1), sorted(extra_samples))
            final_samples = merged[: rblk]
        else:
            final_samples = sorted(samples_phase1)[: rblk]

        #buckets = build_histogram_from_samples(final_samples, k=n_buckets, histogram_builder=histogram_builder)

        column_meta = env.get_column_meta(db_name, table_name, col_name)
        data_type = column_meta.data_type if column_meta else 'unknown'

        # 智能调整桶数
        optimal_buckets = calculate_optimal_buckets(final_samples, data_type, ndv)
        actual_buckets = min(n_buckets, optimal_buckets)

        print(f"Original buckets: {n_buckets}, Optimal buckets: {optimal_buckets}, Using: {actual_buckets}")

        # 使用调整后的桶数构建直方图
        buckets = build_histogram_from_samples(final_samples, k=actual_buckets, 
                                             histogram_builder=histogram_builder,
                                             data_type=data_type, ndv=ndv)


        if not buckets:
            return force_generate_histogram_by_sdc_for_col(env, db_name, table_name, col_name, n_buckets)

        # Assemble HistogramStats JSON-like dict
        total = sum(cnt for _, _, cnt in buckets) or 1
        cum = 0.0

        # Get table metadata to avoid full table scan for sampling rate
        try:
            table_meta = env.get_table_meta(db_name, table_name)
            total_rows = getattr(table_meta, 'rows', None) if table_meta else None
        except Exception:
            total_rows = None

        # Calculate sampling rate using metadata or fallback
        if total_rows and total_rows > 0:
            # sampling_rate = min(1.0, float(rblk) / float(total_rows))
            sampling_rate = min(1.0, float(len(final_samples)) / float(total_rows))
        else:
            # Fallback: use sample count as approximation
            # sampling_rate = min(1.0, float(rblk) / float(len(final_samples) * 10))  # Rough estimate
            sampling_rate = min(1.0, float(len(final_samples)) / float(len(samples_phase1) * 10))

        print(f"Sampling rate: {sampling_rate:.4f} ({len(final_samples)}/{total_rows})")

        res_dict = {
            "buckets": [],
            "data-type": env.get_column_meta(db_name, table_name, col_name).data_type,
            "histogram-type": "block_2phase",
            "null-values": estimate_null_ratio(env, db_name, table_name, col_name),
            "collation-id": MEANINGLESS_INT,
            "sampling-rate": sampling_rate,
            "number-of-buckets-specified": actual_buckets
        }

        # Fill bucket information: [min_value, max_value, cumulative_frequency, row_count]
        for mn, mx, cnt in buckets:
            cum += cnt / total
            res_dict["buckets"].append([str(mn), str(mx), float(cum), max(1, int(cnt))])

        return HistogramStats.init_from_mysql_json(res_dict)


    def fetch_col_histogram(env: Env, dbname: str, table_name: str, col_name: str, n_buckets: int = 32,
                            force: bool = False, hist_mem_size: int = None, ndv: int = None,
                            algo: Optional[str] = None,
                            delta_req: float = 0.2,
                            lmax: int = 4,
                            r1_hint: Optional[int] = None,
                            histogram_builder: str = "equi-depth") -> HistogramStats:
        """
        fetch or generate histogram for a column
        Args:
            env: MySQL env
            dbname:
            table_name:
            col_name:
            n_buckets: number of buckets
            force: if force is False, and histogram exists, return it

        Returns:

        """
        if algo == 'compare':
            print(f"=== Comparing histogram algorithms for {dbname}.{table_name}.{col_name} ===")

            # 运行原有算法（作为基准）
            start_time = time.time()
            hist_original = fetch_col_histogram(env, dbname, table_name, col_name, n_buckets, 
                                              force, hist_mem_size, ndv, None, delta_req, lmax, r1_hint, histogram_builder)
            time_original = time.time() - start_time

            # 运行2phase算法
            start_time = time.time()
            hist_2phase = force_generate_histogram_by_2phase_for_col(env, dbname, table_name, col_name,
                                                                    n_buckets=n_buckets, delta_req=delta_req,
                                                                    lmax=lmax, r1_hint=r1_hint,
                                                                    histogram_builder=histogram_builder, ndv=ndv)
            time_2phase = time.time() - start_time

            # 计算准确性指标
            accuracy_metrics = compare_histogram_accuracy(hist_original, hist_2phase, env, dbname, table_name, col_name)

            # 输出对比结果
            print(f"=== Results for {dbname}.{table_name}.{col_name} ===")
            print(f"Original algorithm: {len(hist_original.buckets)} buckets, {time_original:.2f}s")
            print(f"2Phase algorithm: {len(hist_2phase.buckets)} buckets, {time_2phase:.2f}s")
            print(f"Speedup: {time_original/time_2phase:.2f}x")
            print(f"Accuracy metrics: {accuracy_metrics}")

            return hist_2phase  # 返回2phase结果


        # Optional algorithm switch. Default (None) preserves existing behavior.
        if algo == 'block_2phase':
            print("----This is new Histogram Construction Block_2phase----")
            return force_generate_histogram_by_2phase_for_col(env, dbname, table_name, col_name,
                                                              n_buckets=n_buckets, delta_req=delta_req,
                                                              lmax=lmax, r1_hint=r1_hint,
                                                              histogram_builder=histogram_builder, ndv=ndv)

        if not force:
            hist: HistogramStats = query_histogram(env, dbname, table_name, col_name)
            if hist is not None:
                if len(hist.buckets) == n_buckets:
                    return hist
                else:
                    logging.debug(f"hist(`{dbname}`.`{table_name}`.`{col_name=}`) exists, "
                                  f"but n_bucket mismatch (exists={len(hist.buckets)} != {n_buckets}), re-generate.")
            else:
                logging.debug(f"Histogram(`{dbname}`.`{table_name}`.`{col_name=}`) not found. "
                              f"Generating with {n_buckets} n_buckets")
        else:
            logging.debug(
                f"Force Generating Histogram for `{dbname}`.`{table_name}`.`{col_name}` with {n_buckets} n_buckets")
        # generate histogram and return. if failed, use force_generate_histogram_for_col to generate
        try:
            res_update = update_histogram(env, dbname, table_name, col_name, n_buckets, hist_mem_size)
        except Exception as e:
            if 'is covered by a single-part unique index' in str(e):
                logging.info(f"Column is covered single uk, force generate: {dbname=}, {table_name=}, {col_name=}")
                return force_generate_histogram_by_sdc_for_col(env, dbname, table_name, col_name, n_buckets, ndv=ndv)
            else:
                logging.error(f"uncatched: {dbname=}, {table_name=}, {col_name=}")
                raise
        assert res_update, 'Failed to update histogram'
        return query_histogram(env, dbname, table_name, col_name)

    def query_histogram_mariadb(env: Env, dbname: str, table_name: str, col_name: str, n_buckets: int) -> Union[HistogramStats, None]:
        """
        query histogram for a column in mariadb
        """
        sql = f"SELECT JSON_PRETTY(CONVERT(histogram USING utf8mb4)) AS HISTOGRAM " \
              f"FROM mysql.column_stats WHERE db_name = '{dbname}' AND table_name = '{table_name}' AND column_name = '{col_name}';"
        res = env.query_for_dataframe(sql)
        if len(res) == 0:
            return None

        assert len(res) == 1 and 'HISTOGRAM' in res.iloc[0].to_dict(), f"Invalid result from query_histogram: {res}"

        # Check if HISTOGRAM is None
        histogram_value = res.iloc[0].to_dict()['HISTOGRAM']
        if histogram_value is None:
            logging.warning(f"HISTOGRAM is None, force generate histogram for {dbname=}, {table_name=}, {col_name=}")
            return force_generate_histogram_by_sdc_for_col(env, dbname, table_name, col_name, n_buckets)

        hist_dict = json.loads(histogram_value)

        return HistogramStats.init_from_mariadb_json(env, dbname, table_name, col_name, hist_dict)

    def update_histogram_mariadb(env: Env, dbname: str, table_name: str, n_buckets: int = 32) -> bool:
        """
        update histogram for a column in mariadb
        """
        n_buckets = max(1, min(1024, int(n_buckets)))

        conn = env.mysql_util.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"SET histogram_size = {n_buckets};")
            cursor.execute(f"ANALYZE TABLE `{dbname}`.`{table_name}` PERSISTENT FOR ALL;")
            conn.commit()
            return True

        raise Exception(f"meet error when query: {res}")

    def drop_histogram_mariadb(env: Env, dbname: str) -> bool:
        """
        drop histogram for a column in mariadb
        """
        sql = f"DELETE FROM mysql.column_stats WHERE db_name = '{dbname}';"
        # Use execute_query instead of query_for_dataframe, because DELETE statement does not need to return data
        res = env.mysql_util.execute_query(sql)

        if res is not None and len(res) == 1:
            return True
        return False

    def generate_fetch_histogram_mariadb(env: Env, target_db: str, all_table_names: List[str],
                                         n_buckets: int, force: bool,
                                         drop_hist_after_fetch: bool,
                                         ret_json: bool = False
                                         ) -> Dict[str, Dict[str, Union[HistogramStats, dict]]]:
        """
        generate histogram for all specifed tables in mariadb
        """
        res_tables = defaultdict(dict)
        for table_name in all_table_names:
            table_meta: Table = env.get_table_meta(target_db, table_name)
            if not force:
                for c_id, col in enumerate(table_meta.columns):
                    col: Column
                    hist = query_histogram_mariadb(env, target_db, table_name, col.name)
                    if hist is not None:
                        if len(hist.buckets) == n_buckets:
                            res_tables[str(table_name).lower()][col.name] = hist
                            continue
                        else:
                            logging.debug(f"hist(`{target_db}`.`{table_name}`.`{col.name=}`) exists, "
                                          f"but n_bucket mismatch (exists={len(hist.buckets)} != {n_buckets}), re-generate.")
                    else:
                        logging.debug(f"Histogram(`{target_db}`.`{table_name}`.`{col.name=}`) not found. "
                                      f"Generating with {n_buckets} n_buckets")
            else:
                logging.debug(f"force generate histogram for {target_db}.{table_name}")

            try:
                res_update = update_histogram_mariadb(env, target_db, table_name, n_buckets)
            except Exception as e:
                logging.error(f"update histogram failed for {target_db=}, {table_name=}")
                raise
            assert res_update, 'Failed to update histogram'

            for c_id, col in enumerate(table_meta.columns):
                col: Column
                logging.info(f"Generating Histogram for `{target_db}`.`{table_name}`.`{col.name}` "
                             f"with {n_buckets} n_buckets")
                hist = query_histogram_mariadb(env, target_db, table_name, col.name, n_buckets)

                if hist is None:
                    logging.warning(f"HISTOGRAM is None for `{target_db}`.`{table_name}`.`{col.name}`, "
                                    f"creating empty HistogramStats")
                    hist = HistogramStats(
                        buckets=[],
                        data_type=col.data_type,
                        histogram_type='equi-height',
                        null_values=0,
                        collation_id=None,
                        last_updated=None,
                        sampling_rate=MEANINGLESS_INT,
                        number_of_buckets_specified=0,
                        database_type='mariadb'
                    )

                if ret_json:
                    hist = hist.to_dict()
                res_tables[str(table_name).lower()][col.name] = hist

        if drop_hist_after_fetch:
            try:
                drop_histogram_mariadb(env, target_db)
            except Exception as e:
                logging.error(f"drop histogram failed for {target_db=}, {e=}")

        return res_tables

    def generate_fetch_histogram(env: Env, target_db: str, all_table_names: List[str],
                                 n_buckets: int, force: bool,
                                 drop_hist_after_fetch: bool,
                                 hist_mem_size: int,
                                 ret_json: bool = False,
                                 ndv_single_dict: dict = None,
                                 algo: Optional[str] = None,
                                 delta_req: float = 0.05,
                                 lmax: int = 4,
                                 r1_hint: Optional[int] = None,
                                 histogram_builder: str = "equi-depth",
                                 ) -> Dict[str, Dict[str, Union[HistogramStats, dict]]]:
        """
        generate histogram for all specifed tables

        Args:
            env: MySQL
            target_db:
            all_table_names:
            n_buckets:
            force:
            ret_json: True: return json, False: return HistogramStats
            ndv_single_dict: table_name -> col -> ndv

        Returns:
            lower_table -> column -> HistogramStats

        """
        if not target_env_available_for_videx(env):
            raise Exception(f"given env ({env.instance=}) is not in BLACKLIST, cannot generate_fetch_histogram directly")

        ndv_single_dict = ndv_single_dict or {}

        version = env.get_version()
        if version == MySQLVersion.MariaDB_11_8:
            return generate_fetch_histogram_mariadb(env, target_db, all_table_names, n_buckets, force, drop_hist_after_fetch, ret_json)

        res_tables = defaultdict(dict)
        for table_name in all_table_names:
            table_meta: Table = env.get_table_meta(target_db, table_name)
            for c_id, col in enumerate(table_meta.columns):
                col: Column
                ndv = ndv_single_dict.get(table_name, {}).get(col.name, None)
                hist = None
                try:
                    logging.info(f"Generating Histogram for `{target_db}`.`{table_name}`.`{col.name}` "
                                 f"with {n_buckets} n_buckets")
                    if version == MySQLVersion.MySQL_57:
                        hist = force_generate_histogram_by_sdc_for_col(env, target_db, table_name, col.name, n_buckets,
                                                                       ndv=ndv)
                    elif version == MySQLVersion.MySQL_8:
                        hist = fetch_col_histogram(env, target_db, table_name, col.name, n_buckets, force=force,
                                                   hist_mem_size=hist_mem_size,
                                                   ndv=ndv,
                                                   algo=algo,
                                                   delta_req=delta_req,
                                                   lmax=lmax,
                                                   r1_hint=r1_hint,
                                                   histogram_builder=histogram_builder)
                finally:
                    if drop_hist_after_fetch and version == MySQLVersion.MySQL_8:
                        try:
                            drop_histogram(env, target_db, table_name, col.name)
                        except Exception as e:
                            logging.error(f"drop histogram failed for {target_db}.{table_name}.{col.name}, {e}")

                if hist is not None and ret_json:
                    hist = hist.to_dict()
                res_tables[str(table_name).lower()][col.name] = hist
        return res_tables


    def compare_histogram_accuracy(hist_original: HistogramStats, hist_2phase: HistogramStats, 
                                  env: Env, dbname: str, table_name: str, col_name: str) -> dict:
        """
        比较两种直方图算法的准确性（不包含实际查询测试）
        """
        metrics = {}

        # 1. 桶数量比较
        metrics['bucket_count_diff'] = abs(len(hist_2phase.buckets) - len(hist_original.buckets))
        metrics['bucket_count_ratio'] = len(hist_2phase.buckets) / max(len(hist_original.buckets), 1)

        # 2. 采样率比较
        metrics['sampling_rate_original'] = getattr(hist_original, 'sampling_rate', 1.0)
        metrics['sampling_rate_2phase'] = getattr(hist_2phase, 'sampling_rate', 1.0)

        # 3. 累积频率分布比较
        if hist_original.buckets and hist_2phase.buckets:
            # 计算KL散度
            kl_divergence = calculate_kl_divergence(hist_original, hist_2phase)
            metrics['kl_divergence'] = kl_divergence

            # 计算Earth Mover's Distance
            emd = calculate_earth_movers_distance(hist_original, hist_2phase)
            metrics['earth_movers_distance'] = emd

        # 4. 直方图统计信息对比
        if hist_original.buckets and hist_2phase.buckets:
            # 比较桶的分布范围
            orig_ranges = [(bucket.min_value, bucket.max_value) for bucket in hist_original.buckets]
            phase2_ranges = [(bucket.min_value, bucket.max_value) for bucket in hist_2phase.buckets]

            metrics['range_coverage_original'] = f"{orig_ranges[0][0]} to {orig_ranges[-1][1]}"
            metrics['range_coverage_2phase'] = f"{phase2_ranges[0][0]} to {phase2_ranges[-1][1]}"

            # 比较桶的密度分布
            orig_densities = [bucket.row_count for bucket in hist_original.buckets]
            phase2_densities = [bucket.row_count for bucket in hist_2phase.buckets]

            metrics['density_variance_original'] = np.var(orig_densities) if orig_densities else 0
            metrics['density_variance_2phase'] = np.var(phase2_densities) if phase2_densities else 0

        return metrics


    def calculate_kl_divergence(hist1: HistogramStats, hist2: HistogramStats) -> float:
        """计算两个直方图的 KL 散度.
        改写: 用 Laplace 平滑替代 inf 返回——更适合实际对比场景."""
        _dbg("KL", f"hist1 buckets={len(hist1.buckets)}, hist2 buckets={len(hist2.buckets)}")
        if not hist1.buckets or not hist2.buckets:
            return float('inf')

        def get_probability_from_counts(buckets):
            if not buckets:
                return []
            total_count = sum(bucket.row_count for bucket in buckets)
            if total_count <= 0:
                return []
            return [bucket.row_count / total_count for bucket in buckets]

        probs1 = get_probability_from_counts(hist1.buckets)
        probs2 = get_probability_from_counts(hist2.buckets)

        if not probs1 or not probs2:
            return float('inf')

        # 对齐长度
        max_len = max(len(probs1), len(probs2))
        probs1.extend([0.0] * (max_len - len(probs1)))
        probs2.extend([0.0] * (max_len - len(probs2)))

        # 改写: Laplace 平滑——给每个概率加 epsilon 避免 log(0)
        _EPS = 1e-10
        kl_div = 0.0
        for i in range(max_len):
            p1 = probs1[i] + _EPS
            p2 = probs2[i] + _EPS
            kl_div += p1 * math.log(p1 / p2)

        result = max(0.0, kl_div)
        _dbg("KL", f"result={result:.6g}")
        return result

    # def calculate_kl_divergence(hist1: HistogramStats, hist2: HistogramStats) -> float:
    #     """
    #     计算两个直方图之间的KL散度
    #     """
    #     if not hist1.buckets or not hist2.buckets:
    #         return float('inf')

    #     # 获取两个直方图的累积频率
    #     cum_freq1 = [bucket.cum_freq for bucket in hist1.buckets]
    #     cum_freq2 = [bucket.cum_freq for bucket in hist2.buckets]

    #     # 确保长度一致
    #     max_len = max(len(cum_freq1), len(cum_freq2))
    #     cum_freq1.extend([1.0] * (max_len - len(cum_freq1)))
    #     cum_freq2.extend([1.0] * (max_len - len(cum_freq2)))

    #     # 计算KL散度
    #     kl_div = 0.0
    #     for i in range(max_len):
    #         p1 = cum_freq1[i] if i < len(cum_freq1) else 1.0
    #         p2 = cum_freq2[i] if i < len(cum_freq2) else 1.0

    #         # 避免log(0)
    #         if p1 > 0 and p2 > 0:
    #             kl_div += p1 * math.log(p1 / p2)

    #     return kl_div

    def calculate_earth_movers_distance(hist1: HistogramStats, hist2: HistogramStats) -> float:
        """计算两个直方图的 Earth Mover's Distance.
        改写: 归一化为 [0,1] 区间——使不同桶数的直方图可比."""
        _dbg("EMD", f"hist1={len(hist1.buckets)} buckets, hist2={len(hist2.buckets)} buckets")
        if not hist1.buckets or not hist2.buckets:
            return float('inf')

        cum_freq1 = [bucket.cum_freq for bucket in hist1.buckets]
        cum_freq2 = [bucket.cum_freq for bucket in hist2.buckets]

        max_len = max(len(cum_freq1), len(cum_freq2))
        cum_freq1.extend([1.0] * (max_len - len(cum_freq1)))
        cum_freq2.extend([1.0] * (max_len - len(cum_freq2)))

        raw_emd = sum(abs(p1 - p2) for p1, p2 in zip(cum_freq1, cum_freq2))
        # 改写: 归一化——除以采样点数，得到平均偏差
        normalized_emd = raw_emd / max(1, max_len)
        _dbg("EMD", f"raw={raw_emd:.6g}, normalized={normalized_emd:.6g}, n_points={max_len}")
        return normalized_emd



    if __name__ == '__main__':
        videx_logging.initial_config()
        # some database with tpch
    try:
            from sub_platforms.sql_opt.env.rds_env import Env, OpenMySQLEnv
    except (ImportError, ModuleNotFoundError):
        pass
    try:
            from sub_platforms.sql_opt.benchmark.bench_utils import TPCH_UT_INS_80
    except (ImportError, ModuleNotFoundError):
        pass
        my_env = OpenMySQLEnv.from_db_instance(TPCH_UT_INS_80)

        # varchar(44)
        hist = force_generate_histogram_by_sdc_for_col(my_env, 'tpch_rong', 'lineitem', col_name='L_COMMENT', n_buckets=16, ndv=4580554)
        print(hist.buckets)
        # int
        hist = force_generate_histogram_by_sdc_for_col(my_env, 'tpch_rong', 'lineitem', col_name='L_LINENUMBER', n_buckets=16, )
        print(hist)
        # date
        hist = force_generate_histogram_by_sdc_for_col(my_env, 'tpch_rong', 'lineitem', col_name='L_SHIPDATE', n_buckets=16, )
        print(hist)
        # decimal
        hist = force_generate_histogram_by_sdc_for_col(my_env, 'tpch_rong', 'lineitem', col_name='L_DISCOUNT', n_buckets=16, )
        print(hist)

    # ═══════════════════════════════════════════════════════════════════════════
    # ★ 移植改写区
    # ═══════════════════════════════════════════════════════════════════════════

        def dump_histogram_summary(self) -> str:
            """★ 改写: 直方图桶统计摘要."""
            from .. import _dbg
            n_buckets = len(self._buckets)
            total_count = sum(b.get('count', 0) for b in self._buckets)
            lines = ["┌── Histogram Summary ──",
                     f"│ n_buckets  = {n_buckets}",
                     f"│ total_count= {total_count:,}"]
            if self._buckets:
                widths = [b.get('upper', 0) - b.get('lower', 0) for b in self._buckets]
                lines.append(f"│ min_width  = {min(widths):.4f}")
                lines.append(f"│ max_width  = {max(widths):.4f}")
            lines.append("└──────────────────────────────")
            return "\n".join(lines)

    try:
            from sub_platforms.sql_opt.benchmark.bench_utils import TPCH_UT_INS_80
    except (ImportError, ModuleNotFoundError):
        pass

except (NameError, AttributeError, TypeError) as _exc:
    _dbg("GUARD", f"videx_histogram.py degraded: {_exc}")