# 本地 OCR 服务 API

在本机用 PP-OCRv6（MLX，Apple Silicon）识别一页图片，返回每行文字、行框，以及每个字的位置。接口与 PaddleOCR 云端的 **异步任务接口**（`/api/v2/ocr/jobs`）保持一致，已有的云端客户端只需要改地址，其余流程和解析代码可以不动。

## 1. 基本信息

| 项目 | 说明 |
|---|---|
| 默认地址 | `http://127.0.0.1:8766` |
| 认证 | 无。`Authorization` 头会被忽略，不需要 API key |
| 监听范围 | 默认只监听本机（`127.0.0.1`），不要直接暴露到局域网或公网 |
| 并发 | 识别是串行的：同一时刻只处理一张图，其余请求排队 |
| 平台 | 仅 Apple Silicon 的 macOS |

流程三步：**提交图片 → 查询任务状态 → 下载结果**。本服务在提交请求里就同步完成识别，所以提交返回时任务已经是 `done`，查询一次即可拿到结果地址。

```
POST /api/v2/ocr/jobs                    →  {"data":{"jobId":"…"}}
GET  /api/v2/ocr/jobs/{jobId}            →  {"data":{"state":"done","resultUrl":{"jsonUrl":"…"}}}
GET  /api/v2/ocr/jobs/{jobId}/result.jsonl →  识别结果（JSONL）
```

## 2. 接口

### 2.1 提交识别 `POST /api/v2/ocr/jobs`

`multipart/form-data`，字段如下：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | 是 | 页面图片。支持 OpenCV 能解码的格式（PNG、JPEG、BMP、TIFF、WebP 等）。PDF 不能直接提交，需要先渲染成图片 |
| `model` | 否 | 为兼容云端接口而保留，**会被忽略**。实际使用的模型由服务端环境变量决定（默认 medium） |
| `optionalPayload` | 否 | JSON 字符串，见下表。不是合法 JSON 返回 400 |

`optionalPayload` 里目前只有一个字段有效：

| 键 | 默认 | 说明 |
|---|---|---|
| `returnWordBox` | `false` | 为 `true` 时结果里带 `text_word` 和 `text_word_boxes`（单字位置，见 3.2） |

云端的 `useDocOrientationClassify`、`useDocUnwarping`、`useTextlineOrientation` 等其他键会被忽略，本服务不做文档方向矫正和去畸变。

成功响应（`200`）：

```json
{"data": {"jobId": "ca0db34316594eb1a56b112e63cc2e00"}}
```

### 2.2 查询任务 `GET /api/v2/ocr/jobs/{jobId}`

```json
{
  "data": {
    "state": "done",
    "resultUrl": {"jsonUrl": "http://127.0.0.1:8766/api/v2/ocr/jobs/ca0d…/result.jsonl"}
  }
}
```

`state` 恒为 `done`（没有 `pending`、`running`、`failed`）。识别出错时，错误直接在提交请求里返回（见第 4 节）。`jobId` 不存在返回 404。

### 2.3 下载结果 `GET /api/v2/ocr/jobs/{jobId}/result.jsonl`

`Content-Type: application/x-ndjson`。内容只有一行 JSON，结构见第 3 节。`jobId` 不存在返回 404。

### 2.4 健康检查 `GET /healthz`

```json
{"loaded": false, "idle_seconds": 300.0}
```

`loaded` 表示模型当前是否在内存里。可以用它判断服务是否已启动，**检查本身不会触发模型加载**。

## 3. 结果格式

### 3.1 整体结构

```json
{
  "result": {
    "ocrResults": [
      {
        "prunedResult": {
          "width": 1280,
          "height": 720,
          "rec_texts": ["Please lower your volume", "when you pass by"],
          "rec_scores": [0.9652, 0.9844],
          "rec_boxes": [[13.0, 12.0, 258.0, 54.0], [33.0, 47.0, 210.0, 83.0]],
          "rec_polys": [[[13, 12], [258, 17], [257, 54], [13, 49]], "…"],
          "dt_polys": "与 rec_polys 相同",
          "dt_scores": [0.6762, 0.7702],
          "text_word": [["P", "l", "e", "…"], "…"],
          "text_word_boxes": [[[37.55, 12.5, 43.82, 49.63], "…"], "…"],
          "input_path": null,
          "page_index": null,
          "text_rec_score_thresh": 0.0
        }
      }
    ]
  }
}
```

`ocrResults` 固定只有一个元素（一次提交对应一页图片）。

### 3.2 字段说明

**坐标系**：全部是**输入图片的像素坐标，原点在左上角**，x 向右、y 向下。没有做任何缩放，直接用 `width`、`height` 就能归一化。

| 字段 | 类型 | 说明 |
|---|---|---|
| `width`、`height` | 整数 | 提交的图片尺寸（像素） |
| `rec_texts` | 字符串数组 | 每行的识别文字 |
| `rec_scores` | 小数数组 | 每行的识别置信度，0~1 |
| `rec_boxes` | `[x1,y1,x2,y2]` 数组 | 每行的外接矩形（左上角和右下角） |
| `rec_polys`、`dt_polys` | 四点数组 | 每行的四边形，顺序为左上、右上、右下、左下。文字倾斜时比 `rec_boxes` 更贴合。两个字段内容相同 |
| `dt_scores` | 小数数组 | 每行的检测置信度，0~1 |
| `text_word` | 字符串数组的数组 | 仅 `returnWordBox` 为真时存在。第 i 行拆成单个字符，**拼起来恒等于 `rec_texts[i]`** |
| `text_word_boxes` | 框数组的数组 | 仅 `returnWordBox` 为真时存在。与 `text_word` 一一对应，每个框是 `[x1,y1,x2,y2]` |
| `input_path`、`page_index`、`text_rec_score_thresh` | — | 为与云端格式一致而保留，固定为 `null`、`null`、`0.0`，不用理会 |

以上数组按下标一一对应：`rec_texts[i]`、`rec_scores[i]`、`rec_boxes[i]`、`rec_polys[i]`、`text_word[i]` 描述的是同一行。

行的顺序是服务端的阅读顺序（大致自上而下、同一行内自左向右）。多栏或复杂版面请接入方自行按几何位置重新排序和分组。

置信度低于 0.5 的行会被丢弃，不会出现在结果里。一页没有识别到任何文字时，各数组都是空数组，不是错误。

### 3.3 单字框的使用注意

`text_word_boxes` 中每个框是以该字**中心位置**为依据算出来的：

- **中心是可用的**，来自识别模型对每个字的峰值位置，抽样目检与字形位置基本对齐，但有量化误差，没有做过逐字的精度统计。选字、点击定位、按字拆分请用框的中心 `(x1+x2)/2`。
- **宽度不可靠**。每个框的宽度只是识别模型的一个时间步（图片上通常几个像素），不等于字的实际宽度，窄字符和宽字符的框宽度一样。**不要用框宽去画字的底色或计算字宽**。
- 相邻两个字的分界，建议取相邻中心的中点。
- 竖排文字（高宽比不小于 1.5 的行）也会返回，字序沿竖直方向从上到下，框的 `y` 方向是不同字的位置。
- 行首的符号（比如项目符号 `•`）如果没被识别，那一段区域不会有对应的字，因此第一个字的位置可能离行框左边较远，这是正常的。

### 3.4 示例：把字框换算成归一化坐标

```python
import json

# result.jsonl 是 2.3 下载到的结果；也可以直接用第 5 节 recognize() 返回的 page
line = json.loads(open("result.jsonl").readline())
page = line["result"]["ocrResults"][0]["prunedResult"]
w, h = page["width"], page["height"]
for text, box in zip(page["rec_texts"], page["rec_boxes"]):
    x1, y1, x2, y2 = box
    print(text, (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h))   # 归一化的 x, y, 宽, 高
```

## 4. 错误

错误响应为 JSON：`{"detail": "说明文字"}`。

| 状态码 | 出现位置 | 原因 |
|---|---|---|
| 400 | 提交 | 图片无法解码（`cannot decode image`），或 `optionalPayload` 不是合法 JSON |
| 404 | 查询、下载 | 任务不存在（`unknown job`）。任务结果只保存在内存里，**最多保留最近 64 个**，服务重启后全部丢失 |
| 422 | 提交 | 缺少 `file` 字段等参数校验错误 |
| 500 | 提交 | 识别过程出错（例如模型加载失败），详见服务端日志 |

建议提交成功后立即取结果，不要长期保存 `jobId`。

## 5. 调用示例

curl：

```bash
curl -s -F model=PP-OCRv6 \
  -F 'optionalPayload={"returnWordBox":true}' \
  -F file=@page.png \
  http://127.0.0.1:8766/api/v2/ocr/jobs
# {"data":{"jobId":"…"}}

curl -s http://127.0.0.1:8766/api/v2/ocr/jobs/<jobId>
curl -s http://127.0.0.1:8766/api/v2/ocr/jobs/<jobId>/result.jsonl
```

Python（`httpx`）：

```python
import json
import httpx

BASE = "http://127.0.0.1:8766"

def recognize(png_path: str, word_boxes: bool = True) -> dict:
    with httpx.Client(timeout=120) as client:
        with open(png_path, "rb") as f:
            r = client.post(
                f"{BASE}/api/v2/ocr/jobs",
                data={"model": "PP-OCRv6",
                      "optionalPayload": json.dumps({"returnWordBox": word_boxes})},
                files={"file": ("page.png", f, "image/png")},
            )
        r.raise_for_status()
        job_id = r.json()["data"]["jobId"]

        status = client.get(f"{BASE}/api/v2/ocr/jobs/{job_id}").json()["data"]
        assert status["state"] == "done"
        body = client.get(status["resultUrl"]["jsonUrl"]).text
    return json.loads(body.splitlines()[0])["result"]["ocrResults"][0]["prunedResult"]
```

## 6. 接入云端客户端时的改动清单

已有云端客户端（例如 UniReader 的 `PaddleOCR.swift`）接入本服务，通常只需要：

1. 把任务地址改成 `http://127.0.0.1:8766/api/v2/ocr/jobs`。
2. 不再要求 API key，`Authorization` 头可以不发，发了也不会报错。
3. 轮询逻辑可以保留。第一次查询就是 `done`。
4. **请求超时设宽松（建议不少于 60 秒）**，原因见第 7 节。
5. 结果解析（`rec_texts`、`rec_boxes`、`text_word`、`text_word_boxes`、`width`、`height`）不用改。
6. 缓存键里的识别引擎标识要和云端区分开，两边的识别结果不会完全一致，混用同一个标识会让缓存里的结果来源不明。

## 7. 性能与资源

- **首次请求较慢**：模型是按需加载的，服务启动时不占用模型内存。第一次提交会包含模型加载时间，请把超时设宽松。之后的识别只需要推理时间。
- **空闲自动卸载**：连续 300 秒（默认）没有请求，模型会从内存卸载；下一次请求再加载，所以空闲之后的第一次请求会再慢一次。
- **内存**：加载后进程约占 0.8GB；卸载后会降到约 0.6GB 而不是回到启动时的约 0.1GB，这部分是运行库自身的占用，进程不退出就不会还回去。
- 请求之间严格串行，识别期间新请求会排队等待，不会并行加速。

## 8. 服务端配置

通过环境变量配置，使用 `./ocr-service.sh install` 安装为后台服务时，这些值在安装时写入并固定，修改后需要重新 `install`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `OCR_HOST` | `127.0.0.1` | 监听地址 |
| `OCR_PORT` | `8766` | 监听端口 |
| `OCR_VARIANT` | `medium` | 模型规格：`tiny`、`small`、`medium`。越大越准也越慢 |
| `OCR_IDLE_SECONDS` | `300` | 空闲多久后卸载模型；`0` 表示永不卸载 |

## 9. 协议

本服务以 GPL-3.0 发布，完整文本见仓库根目录的 `LICENSE`。通过 HTTP 调用本服务的程序是另一个独立进程，一般不视为本服务的衍生作品；但如果你把本服务的代码复制进自己的程序里，则需要遵守 GPL-3.0。这不是法律意见，有疑问请咨询专业人士。
