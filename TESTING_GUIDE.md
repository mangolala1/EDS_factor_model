# 测试指南 - 验证修复效果

## 🎯 测试目标

验证以下修复是否生效：
1. ✅ **UnboundLocalError 修复** - `quarter_enterprise_value` 已正确初始化
2. ✅ **Market Value 加载优化** - 从 281 秒降至 ~16 秒
3. ✅ **NaN 警告消除** - 不再出现 "All-NaN slice" 警告
4. ✅ **并行 Worker 优化** - 使用 1-2 个 worker 避免 I/O 争用

---

## 📋 测试步骤

### 方法 1：快速测试单个 Quarter（推荐，~5-10 分钟）

```powershell
# 在项目根目录运行
python test_single_quarter.py
```

**观察要点**：
- ✅ **Market Value 加载时间**：应该显示 `~16s` 而不是 `281s`
- ✅ **无错误**：不应该出现 `UnboundLocalError`
- ✅ **无警告**：不应该出现 "All-NaN slice" 或 "empty slice" 警告
- ✅ **Worker 数量**：日志应该显示 "Using 1 parallel workers"

**预期输出示例**：
```
[14:40:28] 2020Q1: Loading market value...
[14:40:44] 2020Q1: Loaded 3,133,881 market value rows in 16.2s  ← 应该是 ~16s，不是 281s
```

---

### 方法 2：测试完整工作流（2020Q1-Q4，~30-60 分钟）

```powershell
# 运行完整主程序，但限制日期范围
python src/main.py --start-date 2020-01-01 --end-date 2020-12-31
```

或者直接修改 `src/main.py` 中的日期范围：

```python
if __name__ == '__main__':
    main(
        start_date='2020-01-01',
        end_date='2020-12-31',  # 只测试 2020 年
        max_workers=1  # 使用 1 个 worker
    )
```

**观察要点**：
- ✅ **每个 Quarter 的 Market Value 加载时间**：都应该在 16-30 秒左右
- ✅ **无 I/O 争用**：使用 1 个 worker，不应该出现长时间卡顿
- ✅ **处理进度**：应该能看到稳定的进度更新

---

### 方法 3：检查输出文件

处理完成后，检查输出文件：

```powershell
# 检查输出文件是否存在
dir results\exposures_*.parquet

# 检查文件大小（应该不为空）
python -c "import pandas as pd; df = pd.read_parquet('results/exposures_2020Q1.parquet'); print(f'Rows: {len(df)}, Columns: {len(df.columns)}')"
```

---

## 🔍 关键指标对比

### 修复前（4 个 Worker）
- Market Value 加载：**281 秒/quarter**
- I/O 争用：严重（4 个进程同时读同一文件）
- 警告：大量 "All-NaN slice" 警告
- 错误：`UnboundLocalError: quarter_enterprise_value`

### 修复后（1-2 个 Worker）
- Market Value 加载：**~16-30 秒/quarter** ✅
- I/O 争用：无（顺序处理或少量并行）
- 警告：无或极少 ✅
- 错误：无 ✅

---

## ⚠️ 如果测试失败

### 问题 1：仍然出现 UnboundLocalError
**检查**：
```python
# 在 src/quarter_processor.py 第 303 行附近
quarter_enterprise_value = pd.DataFrame()  # 应该总是被初始化
```

### 问题 2：Market Value 仍然很慢（>100 秒）
**检查**：
1. 是否使用了列选择？查看日志中是否有 "Only read needed columns"
2. 是否使用了 PyArrow dataset？查看是否有 "Using dataset with column selection"
3. Worker 数量：确保 `max_workers=1` 或 `2`

### 问题 3：仍然有 NaN 警告
**检查**：
```python
# 在 src/quarter_processor.py 第 726-747 行
# 应该看到列级别的 guard 逻辑
for col_idx in range(char_data_clipped.shape[1]):
    valid_mask = np.isfinite(col_data)
    valid_count = valid_mask.sum()
    # ...
```

---

## 📊 性能基准

### 单个 Quarter (2020Q1) 预期时间：
- **Prices 加载**：~180-200 秒
- **Returns 加载**：~120-130 秒
- **Market Value 加载**：~16-20 秒 ✅（修复前：281 秒）
- **Fundamentals Snapshot 初始化**：~0.6 秒
- **Enterprise Value Snapshot 初始化**：~0.1 秒
- **Exchange Rates Snapshot 初始化**：~0.0 秒
- **Buffer 填充**：~250-300 秒（取决于历史数据量）
- **每日处理**：~5-10 秒/天

**总计**：单个 Quarter 约 **10-15 分钟**

### 完整 2020 年（4 个 Quarters）预期时间：
- **串行处理（1 worker）**：~40-60 分钟
- **并行处理（2 workers）**：~30-45 分钟（可能更慢，取决于 I/O）

---

## ✅ 成功标准

测试通过如果：
1. ✅ 没有 `UnboundLocalError`
2. ✅ Market Value 加载时间 < 30 秒/quarter
3. ✅ 没有或极少 "All-NaN slice" 警告
4. ✅ 成功生成输出文件 `results/exposures_2020Q*.parquet`
5. ✅ 输出文件包含预期的列（DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE 等）

---

## 🚀 下一步

如果测试通过：
1. **运行完整历史数据**（2020-01-01 至今）
2. **监控性能**：记录每个阶段的耗时
3. **验证输出**：检查 exposures、factor_returns、specific_returns 等表

如果测试失败：
1. **查看错误日志**：定位具体问题
2. **检查数据文件**：确保 Parquet 文件完整
3. **逐步调试**：先测试更小的日期范围

