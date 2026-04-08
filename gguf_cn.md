原始 URL：<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

# GGUF

GGUF 是一种用来保存模型的文件格式，用来存储使用 GGML 进行推理的模型以及基于 GGML 的执行器。GGUF 是一种二进制格式，目的是用来快速加载和保存模型，并且易于读取。在过去，模型是使用 PyTorch 或者其他框架开发的，然后转换为 GGUF 以便在 GGML 中使用。

它是 GGML 、 GGMF 和 GGJT 的后续文件格式，旨在通过包含加载模型所需的所有信息来确保没有歧义。除此以外，它还具有扩展性，因此可以在不破坏兼容性的前提下想模型中添加新信息。

更多关于 GGUF 背后动机的信息，可以查看 [历史情况](#历史情况) 。

## 规格

GGUF 是一种基于现有的 GGJT 的格式，但是在格式上做了一些调整来让它更有扩展性并且更容易使用。需要具备以下特性：

- 单文件部署：模型可以被很容易的分发和加载，并且不需要任何外部文件来提供附加信息。
- 可扩展：可以向基于 GGML 的执行器中添加新特性 / 可以向 GGUF 模型中添加新信息，而不会破坏对于现有模型的兼容性。
- `mmap` 兼容性：模型可以使用 `mmap` 来加载模型，实现快速加载和保存。
- 易于使用：不管用哪种语言，都可以使用少量的代码轻松加载和保存模型，而无需外部库。
- 信息完整：加载模型需要的全部信息都包含在模型文件中，用户不需要其他信息。

GGJT 和 GGUF 的主要区别就是，GGJT 使用键值（Key-Value）结构来存储超参数（现在被称为元数据），而不是使用没有类型化的值列表。这使得可以在不破坏与现有模型的兼容性的情况下添加新的元数据，并且可以使用可能有助于推理和识别模型来标注模型。

### GGUF 命名规则

GGUF 遵守 `<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf` 这样的命名规则，其中每个组成部分如果存在就使用一个 `-` 分割。最终目的是让人类看一眼就能很容易获得一个模型最重要的细节。由于现有的 GGUF 文件名的多样性，这种命名规则并不打算完全可解析。

这些组成部分包括：
1. **BaseName**：一个模型基本类型或者模型架构的描述性名称。
    - 可以从 GGUF 的元数据 `general.basename` 中得到，只需要将空格换成 `-` 。
2. **SizeLabel**：参数权重等级（对于排行榜很有用）展现为 `<expertCount>x<count><scale-prefix>`
    - 如果可用，可以从 GGUF 的元数据 `general.size_label` ，如果缺失则计算出来。
    - 计数中支持四舍五入的小数点，使用单个字母前缀来辅助如下的浮点指数计数：
      - `Q`：万亿亿个参数。
      - `T`：万亿个参数。
      - `B`：十亿个参数。
      - `M`：百万个参数。
      - `K`：千个参数。
    - 可以按需要添加额外的 `-<attributes><count><scale-prefix>` 来表示其他感兴趣的属性
3. **FineTune**：模型微调目标的一个描述性名称（例如聊天、指令等等）
    - 可以从 GGUF 的元数据 `general.finetune` 中得到，只需要将空格替换成 `-` 。
4. **Version**：（可选） 表示模型的版本号，格式为 `v<Major>.<Minor>`
    - 如果版本号缺失，就假设 `1.0` (第一个公开发布)
    - 可以从 GGUF 的元数据 `general.version` 中得到
5. **Encoding**：表示模型应用的权重编码方案。而内容、类型混合和排列方式由用户的代码决定，并且可以按照项目需求改变。
6. **Type**：表示 GGUF 文件的类型，及其预期用途
    - 如果缺失，那么这个文件默认是一个典型的 GGUF 张量模型文件。
    - `LoRA`： GGUF 文件是 LoRa 适配器。
    - `vocab`：仅包含词汇数据和元数据的 GGUF 文件。
7. **Shard**：（可选） 表示模型已经被拆分成多个分片，格式为 `<ShardNum>-of-<ShardTotal>` 。
    - *ShardNum*：这个模型中的分片位置。必须使用 0 填充为 5 个数字。
      - 分片数必须从 `00001` 开始递增（例如第一个分片通常从 `00001-of-XXXXX` 开始而不是 `00001-of-XXXXX` ）。
    - *ShardTotal*：这个模型中的分片总数。必须使用 0 填充为 5 个数字。


#### 验证上述的命名规则

所有的模型文件至少应该包含 BaseName 、 SizeLabel 和 Version ，以便验证一个文件是否符合 GGUF 的命名规范。例如，如果省略了 Version ，就很容易把 Encoding 误认为是 FineTune 。

为了验证，你可以使用这个正则表达式 `^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$` ，它可以检查你是否至少按照正确的顺序提供了 BaseName 、 SizeLabel 和 Version 。

例如：

  * `Mixtral-8x7B-v0.1-KQ2.gguf`:
    - 模型名称：Mixtral
    - 专家数量：8
    - 参数数量：7B
    - 版本：v0.1
    - 权重编码方案：KQ2

  * `Hermes-2-Pro-Llama-3-8B-F16.gguf`:
    - 模型名称：Hermes 2 Pro Llama 3
    - 专家数量：0
    - 参数数量：8B
    - 版本：v1.0
    - 权重编码方案：F16
    - 分片：N/A

  * `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf`
    - 模型名称：Grok
    - 专家数量：0
    - 参数数量：100B
    - 版本：v1.0
    - 权重编码方案：Q4_0
    - 分片：总共 9 个分片中的第 3 个

<details><summary>Node.js 正则表达式方法示例</summary>

```js
#!/usr/bin/env node
const ggufRegex = /^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$/;

function parseGGUFFilename(filename) {
  const match = ggufRegex.exec(filename);
  if (!match)
    return null;
  const {BaseName = null, SizeLabel = null, FineTune = null, Version = "v1.0", Encoding = null, Type = null, Shard = null} = match.groups;
  return {BaseName: BaseName, SizeLabel: SizeLabel, FineTune: FineTune, Version: Version, Encoding: Encoding, Type: Type, Shard: Shard};
}

const testCases = [
  {filename: 'Mixtral-8x7B-v0.1-KQ2.gguf',                         expected: { BaseName: 'Mixtral',              SizeLabel: '8x7B',     FineTune: null, Version: 'v0.1',   Encoding: 'KQ2',  Type: null, Shard: null}},
  {filename: 'Grok-100B-v1.0-Q4_0-00003-of-00009.gguf',            expected: { BaseName: 'Grok',                 SizeLabel: '100B',     FineTune: null, Version: 'v1.0',   Encoding: 'Q4_0', Type: null, Shard: "00003-of-00009"}},
  {filename: 'Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf',              expected: { BaseName: 'Hermes-2-Pro-Llama-3', SizeLabel: '8B', FineTune: null, Version: 'v1.0',   Encoding: 'F16',  Type: null, Shard: null}},
  {filename: 'Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf', expected: { BaseName: 'Phi-3-mini',   SizeLabel: '3.8B-ContextLength4k', FineTune: 'instruct', Version: 'v1.0',   Encoding: null,  Type: null, Shard: null}},
  {filename: 'not-a-known-arrangement.gguf',                       expected: null},
];

testCases.forEach(({ filename, expected }) => {
  const result = parseGGUFFilename(filename);
  const passed = JSON.stringify(result) === JSON.stringify(expected);
  console.log(`${filename}: ${passed ? "PASS" : "FAIL"}`);
  if (!passed) {
      console.log(result);
      console.log(expected);
  }
});
```

</details>


### 文件结构

![文件结构](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)
*图表由 [@mishig25](https://github.com/mishig25) 绘制（ GGUF v3 ）*

GGUF 的文件结构如下。它们使用在 `general.alignment` 元数据字段中指定的全局对齐，以下称为 `ALIGNMENT` 。在需要时，文件使用 `0x00` 字节填充，使它达到 `general.alignment` 的下一个倍数。

除非专门说明，否则字段（包含数组）将按照顺序写入，不进行对齐。

模型默认采用 [小端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) 。它们可以提供一个 [大端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) 的版本，以便在 [大端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) 的计算机上使用；在这种情况下，所有值（包含元数据的值和张量）也会采用 [大端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) 。在写这个文档的时候，还没有方法决定模型是否是 [大端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) ；这个问题可能会在未来的版本中解决。如果没有提供其他信息，那么假设模型为 [小端序](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F) 。

<details><summary>代码</summary>

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

</details>

## 标准化的键值对（Key-Value）

以下键值对是标准化的。这个列表在未来可能会随着更多用例被发现而扩大。尽可能使用与原始模型定义相同的名称，以便更容易在两者之间映射。

并非所有的键值对都是必需的，但是建议全部添加。必需的键使用粗体。对于省略的键值对，Reader 应当假设它的值是未知的，并且根据情况使用默认值或者报错。

社区可以创建自己的键值对来携带额外的数据。但是，这些键值对应该使用相关的社区名称做命名空间划分，以免冲突。例如，`rustformers` 社区可以使用 `rustformers.` 作为所有键的前缀。

如果特定的社区键被广泛使用，它可能被提升为一个标准化键。

按照惯例，大多数的计数、长度等值都是 `uint64` ，除非特别指明。这是为了未来能够支持更大的模型。一些模型可能为他们的取值使用 `uint32` ；建议 Reader 两种都支持。

### General

#### 必需

- **`general.architecture: string`**：描述此模型的实现框架。只允许左右小写的 ASCII 以及 `[a-z0-9]+` 字符。已知的值包括：
  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `mamba`
  - `rwkv`
- **`general.quantization_version: uint32`**：量化格式的版本。如果模型没有量化（例如没有量化任何张量）则不需要这个键。如果有任何张量进行了量化，这个键 _必须_ 提供。这与张量本身的量化方案无关；量化的版本可以变更，而不用改变方案的名称（例如量化方案为 Q5_K 并且量化版本为 4 ）。
- **`general.alignment: uint32`**：如上所述，要使用全局比对值。这个值可以因不同的对比方案而异，但必须是 8 的倍数。一些 Writer 可能不会写入对比值。如果对比值 **没有** 指定，就假设是 `32` 。

#### General 元数据

- `general.name: string`：模型的名称。这应该是一个人类可读的名字，并可以用来识别模型。它应该在模型定义的社区中是唯一的。
- `general.author: string`：模型的作者。
- `general.version: string`：模型的版本。
- `general.organization: string`：模型的组织。
- `general.basename: string`：模型的基础名称 / 架构。
- `general.finetune: string`：基础模型已经优化的方向。
- `general.description: string`：格式不限的模型描述，包含其他字段没有覆盖的任何内容
- `general.quantized_by: string`：对模型进行量化的人的姓名
- `general.size_label: string`：模型的大小等级，例如权重和专家的数量。（对排行榜很有用）
- `general.license: string`：模型的 License，表达为一个 [SPDX License 表达式](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) （例如 `"MIT OR Apache-2.0`）。不要包含任何其他信息，例如 License 的文本或者指向 License 的链接。
- `general.license.name: string`：人类友好的 License 名称。
- `general.license.link: string`： License 的 URL 。
- `general.url: string`：模型主页的 URL 。这可以是一个 GitHub 仓库、一篇论文等。
- `general.doi: string`：Digital Object Identifier （DOI） https://www.doi.org/
- `general.uuid: string`：[Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.repo_url: string`：模型仓库的 URL ，例如一个 GitHub 仓库或者 HuggingFace 仓库
- `general.tags: string[]`：一个标签的列表，搜索引擎或者社交媒体可以用来做搜索关键词
- `general.languages: string[]`：模型可以说的语言。使用 [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) 的两个字符编码
- `general.datasets: string[]`：模型训练使用的数据集的链接或者引用
- `general.file_type: uint32`：一个枚举值，用来描述文件中文件中大多数张量的类型。这是可选的，可以从张量类型中推断出来。
  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` （移除支持）
  - `MOSTLY_Q4_3 = 6` （移除支持）
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### 源元数据

关于这个模型来源的信息。这有助于追踪模型的出处，并在模型被修改后查找原始来源。例如，对于从 GGML 转换来的模型，这些键将会指向转换前的原始模型。

- `general.source.url: string`：模型主页源的 URL 。这可以是 GitHub 仓库、一篇论文等。
- `general.source.doi: string`：源的 Digital Object Identifier （DOI） https://www.doi.org/
- `general.source.uuid: string`：源的 [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.source.repo_url: string`：模型仓库源的 URL ，例如一个 GitHub 仓库或者一个 HuggingFace 仓库。

- `general.base_model.count: uint32`：父模型的数量
- `general.base_model.{id}.name: string`：父模型的名称
- `general.base_model.{id}.author: string`：父模型的作者
- `general.base_model.{id}.version: string`：父模型的版本
- `general.base_model.{id}.organization: string`：父模型的组织
- `general.base_model.{id}.url: string`：父模型源的 URL 。这可以是一个 GitHub 仓库、一篇论文等。
- `general.base_model.{id}.doi: string`：父模型的 Digital Object Identifier （DOI） https://www.doi.org/
- `general.base_model.{id}.uuid: string`：父模型的 [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.base_model.{id}.repo_url: string`：父模型仓库源的 URL ，例如一个 GitHub 仓库或者 HuggingFace 仓库。

### LLM

在下文中，`[llm]` 用来代表特定 LLM 架构的名称。例如，`llama` 代表 LLaMA 、`mpt` 代表 MPT等。如果在一个架构的部分中提及，那么这个架构需要这个模型，但并非所有架构都需要。请参阅相关的部分了解更多信息。

- `[llm].context_length: uint64`：也称为 `n_ctx` ， 是模型训练所用的上下文长度（以词元为单位）。对于大多数架构，这是输入长度的硬性限制。像 RWKV 这样的不依赖于 Transformer 式注意力机制的架构，或恤能够处理更大的输入，但这无法保证。
- `[llm].embedding_length: uint64`：也称为 `n_embd` 。嵌入层的大小。
- `[llm].block_count: uint64`：注意力层+前馈层（即LLM的主体部分）的模块数量。不包括输入层或嵌入层。
- `[llm].feed_forward_length: uint64`：也称为 `n_ff` 。前馈层的长度。
- `[llm].use_parallel_residual: bool`：代表是否应该使用并行残差逻辑。
- `[llm].tensor_data_layout: string`：当一个模型转换成 GGUF 时，张量可能被重新排序以提高性能。这个键描述了张量数据的的布局。这个键是非必须需的；如果没有出现，就假设为 `reference` 。
  - `reference`：张量的排序与原始模型相同
  - 更多的选项可以在架构的相关章节中找到
- `[llm].expert_count: uint32`：MoE 模型中的专家数量（对于 non-MoE 架构可选）
- `[llm].expert_used_count: uint32`：每个词元评估的过程中使用的专家数量（对于 non-MoE 架构可选）

#### Attention 部分

- `[llm].attention.head_count: uint64`：也称为 `n_head` 。 注意力头的数量。
- `[llm].attention.head_count_kv: uint64`：Grouped-Query-Attention 中每个组的头的数量。如果不存在，或者出现了但是与 `[llm].attention.head_count` 相同，那么这个模型不会使用 GQA 。
- `[llm].attention.max_alibi_bias: float32`：ALiBI 的最大偏差值。
- `[llm].attention.clamp_kqv: float32`：值 `C` 用于将 `Q` 、`K` 和 `V` 张量限制在 `[-C, C]` 之间。
- `[llm].attention.layer_norm_epsilon: float32`：层归一化 $\epsilon$ 。
- `[llm].attention.layer_norm_rms_epsilon: float32`：层 RMS 归一化 $\epsilon$.
- `[llm].attention.key_length: uint32`：一个可选的键头的大小， $d_k$ 。如果没有指定，它会取值 `n_embd / n_head` 。
- `[llm].attention.value_length: uint32`：一个可选的值头的大小， $d_v$ 。如果没有指定，它会取值 `n_embd / n_head` 。

#### RoPE

- `[llm].rope.dimension_count: uint64`：RoPE 的旋转维度的数量。
- `[llm].rope.freq_base: float32`：RoPE 的基准频率。

##### 缩放

以下键描述了 RoPE 缩放参数：

- `[llm].rope.scaling.type: string`：可以是 `none` 、 `linear` 或者 `yarn` 。
- `[llm].rope.scaling.factor: float32`：RoPE 的缩放因子，用来调整上下文的长度。
- `[llm].rope.scaling.original_context_length: uint32_t`：基础模型的原始上下文长度。
- `[llm].rope.scaling.finetuned: bool`：如果模型使用了 RoPE 缩放进行了微调，则值为 True 。

注意，旧模型也许没有这些键，但可能可以使用以下这些键代替：

- `[llm].rope.scale_linear: float32`：RoPE 的一个线性的缩放因子，用来调整上下文的长度。

推荐模型尽可能使用新的键，因为它们更具灵活性，并且支持更复杂的缩放方案。执行器需要无限支持这两方面。

#### SSM

- `[llm].ssm.conv_kernel: uint32`：滚动/平移状态的大小。
- `[llm].ssm.inner_size: uint32`：状态的嵌入大小。
- `[llm].ssm.state_size: uint32`：循环状态的大小。
- `[llm].ssm.time_step_rank: uint32`：时间不长的排名。

#### 模型

以下部分描述了每种模型架构的元数据。每个指定的键 _必须_ 存在。

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### 可选

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `Meta AI original pth`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```
- `llama.expert_count`
- `llama.expert_used_count`

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### 可选

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### 可选

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### 可选

- `falcon.tensor_data_layout`：

  - `jploski` （Falcon 的原始 GGML 实现的作者）：

    ```python
    # The original query_key_value tensor contains n_head_kv "kv groups",
    # each consisting of n_head/n_head_kv query weights followed by one key
    # and one value weight (shared by all query heads in the kv group).
    # This layout makes it a big pain to work with in GGML.
    # So we rearrange them here,, so that we have n_head query weights
    # followed by n_head_kv key weights followed by n_head_kv value weights,
    # in contiguous fashion.

    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

##### Mamba

- `mamba.context_length`
- `mamba.embedding_length`
- `mamba.block_count`
- `mamba.ssm.conv_kernel`
- `mamba.ssm.inner_size`
- `mamba.ssm.state_size`
- `mamba.ssm.time_step_rank`
- `mamba.attention.layer_norm_rms_epsilon`

##### RWKV

词汇表的大小与 `head` 矩阵中的行数相同。

- `rwkv.architecture_version: uint32`：目前允许的值只有 4 。版本 5 预期在未来的某些时候出现。
- `rwkv.context_length: uint64`：训练和微调中使用的上下文的长度。RWKV 可以操控比这个限制更大的上下文，但是输出质量可能会糟糕。
- `rwkv.block_count: uint64`
- `rwkv.embedding_length: uint64`
- `rwkv.feed_forward_length: uint64`

##### Whisper

应当假设没有类型定义的键与 `llm.` 键相同。
（例如，`whisper.context_length` 等价于 `llm.context_length` 。）
这是因为他们都是 Transformer 模型。

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
- `whisper.decoder.attention.head_count`

#### 提示词

**TODO**：包含提示词格式和 / 或关于应该如何使用的元数（说明、对话、自动完成的）。

### LoRA

**TODO**：确定 LoRA 需要哪些元数据。可能需要的特性包括：

- 完全匹配一个现有的模型，以免不会被错误应用
- 标记为 LoRA ，这样执行器不会尝试运行它

应该是一个架构，还是应该保留原始模型的细节并添加额外的字段来把它标记成 LoRA ？

### 分词器

以下键用来描述模型的分词器。推荐模型的作者支持尽可能多的分词器，因为这会为支持的执行器提供更好的分词质量。

#### GGML

GGML 支持一个嵌入的词汇表，让模型能够推理，但是使用这个词汇表的分词器实现（例如 `llama.cpp` 的分词器） 的准确性可能低于模型的原始分词器。当一个更准确的分词器可用并且也支持的时候，则应当使用这个分词器。

无法保证模型之间的标准化，但在未来可能会改变。推荐模型的作者尽可能使用一个更标准的分词器。

- `tokenizer.ggml.model: string`：分词器模型的名称。
  - `llama`：Llama 风格的 SentencePiece（从 HF `tokenizer.model` 中提取的词元和打分）
  - `replit`：Replit 风格的 SentencePiece（从 HF `spiece.model` 中提取的词元和打分）
  - `gpt2`：GPT-2 / GPT-NeoX 风格的 BPE（从 HF `tokenizer.json` 中提取的词元）
  - `rwkv`：RWKV 分词器
- `tokenizer.ggml.tokens: array[string]`：模型使用的一个用词元 ID 索引的词元列表。
- `tokenizer.ggml.scores: array[float32]`：如果出现，它代表每个词元的打分 / 概率。如果没有出现，就假设所有的词元都有同样的概率。如果出现，它的长度必须与 `tokens` 有相同的长度和索引。
- `tokenizer.ggml.token_type: array[int32]`：词元类型（1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte）。如果出现，它的长度必须与 `tokens` 有相同的长度和索引。
- `tokenizer.ggml.merges: array[string]`：如果存在，代表分词器的融合。如果不存在，就假设词元是原子化的。
- `tokenizer.ggml.added_tokens: array[string]`：如果出现，代表训练之后添加的词元。

##### 特别的词元

- `tokenizer.ggml.bos_token_id: uint32`：序列标记的开始
- `tokenizer.ggml.eos_token_id: uint32`：序列标记的结束
- `tokenizer.ggml.unknown_token_id: uint32`：未知的词元
- `tokenizer.ggml.separator_token_id: uint32`：分隔的词元
- `tokenizer.ggml.padding_token_id: uint32`：填充的词元

#### Hugging Face

Hugging Face 会维护他们自己的 `tokenizers` 库，支持大量分词器。如果你的执行器使用这个库，它可能可以直接使用模型的分词器。

- `tokenizer.huggingface.json: string`：给定模型的 HF `tokenizer.json` 的完整内容（例如 <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>）。包含功能是为了与直接支持 HF 分词器的执行器兼容。

#### 其他

可以使用其他的分词器，但是他们可能没有标准化。他们可能是适配特定执行器的。当他们在发现或进一步开发之后，他们会记录在这里。

- `tokenizer.rwkv.world: string`：一个 RWKV 世界的分词器，比如 [这个](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt) 。必须原样引入这个文本文件。
- `tokenizer.chat_template : string`：一个 Jinja 模板，它指定模型需要的输入格式。更多细节可以查看：<https://huggingface.co/docs/transformers/main/en/chat_templating>

### 计算图

这是一个未来的扩展，仍然需要讨论，并且可能需要一个新的 GGUF 版本。在写这个文档的时候，主要阻碍是计算图格式的稳定性。

一个 GGML 节点的计算图样例可以包含在模型当中，允许一个执行器不需要提供它自己的架构实现就可以运行模型。这会在执行器之间提供一个一致的体验，并且能支持一个更复杂的架构而不需要执行器实现他们。

## 标准化的张量名称

为了降低复杂度并提高兼容性，建议使用 Transformer 架构的模型为它们的张量使用如下命名规范：

### 基础层

- `AA.weight`
- `AA.bias`

其中 `AA` 可以是：

- `token_embd`：Token 嵌入层
- `pos_embd`：位置嵌入层
- `output_norm`：输出归一化层
- `output`：输出层

### 注意力层和前馈层的模块

- `blk.N.BB.weight`
- `blk.N.BB.bias`

其中 N 表示的是一个层所属的模块编号，而 `BB` 可以是：

- `attn_norm`：注意力归一化层
- `attn_norm_2`：注意力归一化层
- `attn_qkv`：注意力 查询-键-值 层
- `attn_q`：注意力查询层
- `attn_k`：注意力键层
- `attn_v`：注意力值层
- `attn_output`：注意力输出层

- `ffn_norm`：前馈网络归一化层
- `ffn_up`：前馈网络 Up 层
- `ffn_gate`：前馈网络 Gate 层
- `ffn_down`：前馈网络 Down 层
- `ffn_gate_inp`：MoE 模型中的前馈网络的专家路由层 
- `ffn_gate_exp`：MoE 模型中每个专家的前馈网络 Gate 层
- `ffn_down_exp`：MoE 模型中每个专家的前馈网络 Down 层
- `ffn_up_exp`：MoE 模型中每个专家的前馈网络 Up 层

- `ssm_in`：状态空间模型的输入投影层
- `ssm_conv1d`：状态空间模型的滚动/平移层
- `ssm_x`：状态空间模型的选择性参数化层
- `ssm_a`：状态空间模型的状态压缩层
- `ssm_d`：状态空间模型的跳过连接层
- `ssm_dt`：状态空间模型的时间不长层
- `ssm_out`：状态空间模型的输出投影层

## 历史版本

这个文档在积极更新，来描述元数据的当前状态，但这些变更不会在 Commit 外部追中到。

但是，这个格式 _自身_ 已经改变了。以下部分描述了格式自身的变更。

### v3

增加了对 big-endian 的支持。

### v2

最大可计算的值（长度等）从 `uint32` 变成了 `uint64`，以便未来能够支持更大的模型。

### v1

初始版本。

## 历史情况

以下信息仅供参考，但并不是理解文档的其余部分所必须的。

### 总览

目前，有三种常用的 GGML 文件格式：

- **GGML** （未版本化的）：极限格式，没有版本控制或者对齐方式。
- **GGMF** （版本化的）：与 GGML 相同，但带有版本控制。只存在一个版本。
- **GGJT**：张量对齐，以便与需要对齐的 `mmap` 方法一同使用。v1 、v2 和 v3 版本是完全相同的，但是后面的几个版本使用了不同的量化方案，与前面的版本不兼容。

GGML 主要被 `ggml` 中的样例使用，而 GGJT 被 `llama.cpp` 模型使用。其他执行器可能使用了三种格式中的任意一种，但这不是 “官方” 支持的。

这些格式具有相同的基本结构：

- 一个魔法数字，带有一个可选的版本号
- 模型制定的超参数，包括
  - 关于模型的元数据，例如层的数量、头的数量等。
  - 一个 `ftype` ，它描述了大多数张量的类型，
    - 对于 GGML 文件，量化版本在 `ftype` 中编发，数值除以 1000
- 一个嵌入的词汇表，同时也是一个带有长度前缀的字符串列表。GGMF/GGJT 格式会在字符串旁边嵌入一个 float32 类型的评分。
- 最后，返回一个张量的列表，并带有张量的名称（前面加上长度前缀）、类型以及（对于 GGJT 来说是对齐的）张量数据。

值得注意的事，这种结构既没有指明模型所属的模型架构，也没有提供任何更改超参数结构的灵活性。这意味着添加新的超参数的唯一方法是将它加到列表的末尾，这对现有的模型来说是一个破坏性的变更。

### 不足

不幸的是，在过去的几个月，现有的模型出现了一些问题：

- 没办法确定一个给定的模型适合哪种模型架构，因为没有这个信息
  - 类似的，现有的程序遇到新架构的时候，也没法自动发现
- 添加或者移除任何新的超参数都是一个破坏性的变更，Reader 不使用 [启发式算法](https://zh.wikipedia.org/w/index.php?title=%E5%90%AF%E5%8F%91%E5%BC%8F%E7%AE%97%E6%B3%95) 就不可能发现
- 每种模型架构需要它自己的转换脚本，转换成对应架构的 GGML 变体
- 要维护向后兼容性而不破坏格式的结构需要一些聪明的操作，例如在把量化版本打包到 ftype 中，但这些操作不能保证能被 Reader 或 Writer 获取到，并且在两种格式之间也无法保证一致性。

### 为什么不是其他格式？

有其他几种格式可以使用，但问题包括：

- 需要额外的依赖来加载和保存模型，这在 C 语言环境中会很复杂
- 对 4-bit 量化只有有限的支持或者不支持
- 现有的文化预期（例如模型是目录还是文件）
- 缺少对嵌入式词汇表的支持
- 对于未来的发展方向缺少控制

最终，在可预见的未来，GGUF 仍然可能是必要的，而且，有一个文档齐全并得到所有执行器支持的格式，要好过扭曲现有的格式来适应 GGML 的需求。
