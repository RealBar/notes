# VAE（Variational Autoencoder）工程说明

本目录提供一个可复用的 PyTorch VAE 实现，目标是：

- 用同一套 `VAEBase` + 训练/测试/推理脚本，既能跑 CelebA（人脸生成），也能复用到 MNIST 等其他数据集。
- 模型结构与数据管线解耦：数据集在 `loader/`，模型在 `models/`，训练/测试/推理在根目录脚本。

## 代码结构

目录（省略部分 `__pycache__` 等运行产物）：

```
vae/
  factory.py
  train.py
  test.py
  infer.py
  models/
    base.py
    celeba.py
    mnist.py
    __init__.py
  modules/
    conv_blocks.py
    __init__.py
  loader/
    image_dir.py
    celeba.py
    mnist.py
    __init__.py
  utils/
    checkpoint.py
    device.py
    seed.py
    __init__.py
```

### 顶层脚本

- `train.py`
  - 训练入口：构建数据集/Loader，创建模型，训练循环，保存 checkpoint。
  - 关键点：
    - 通过 `--dataset` 选择数据集（`celeba`/`mnist`）
    - 通过 `--model` 选择模型（默认跟随 `--dataset`）
    - 训练时调用 `VAEBase.compute_loss()` 计算 `recon + beta * KL`
  - 参考：`diffusion/model/vae/train.py:19`

- `test.py`
  - 测试入口：加载 checkpoint，在测试集上计算平均 loss，同时保存：
    - `recon_grid.png`：原图与重建图拼接的网格图
    - `samples_grid.png`：随机采样生成图
  - 参考：`diffusion/model/vae/test.py:16`

- `infer.py`
  - 推理入口：只做随机采样生成，并保存网格图到 `--out`。
  - 参考：`diffusion/model/vae/infer.py:13`

- `factory.py`
  - 模型工厂：统一创建模型实例（`create_model`）与加载 checkpoint（`load_model_checkpoint`）。
  - 通过这个文件把“训练脚本”与“具体模型类”解耦。
  - 参考：`diffusion/model/vae/factory.py:24`

### 模型（models/）

- `models/base.py`
  - 定义通用基类 `VAEBase`：
    - 必须实现：`encode(x) -> (mu, logvar)` 与 `decode(z) -> recon`
    - 提供通用逻辑：
      - `reparameterize(mu, logvar)`
      - `forward(x) -> (recon, mu, logvar)`
      - `compute_loss(recon, x, mu, logvar)` 返回 `VAELoss(loss, recon, kl)`
      - `sample(n)` 与 `reconstruct(x)` 推理接口
  - 损失支持：
    - `recon_loss="mse"`：更适合连续像素（如 CelebA 归一化后）
    - `recon_loss="bce"`：更适合 [0,1] 像素（如 MNIST）
  - 参考：`diffusion/model/vae/models/base.py:22`

- `models/celeba.py`
  - `CelebaVAE`：面向 CelebA 的卷积 VAE。
  - 编码器：多层 `ConvBlock` 下采样，之后用 `AdaptiveAvgPool2d((4,4))` 固定到 4×4，再全连接得到 `mu/logvar`。
  - 解码器：全连接回到 `(C,4,4)`，再多层 `DeconvBlock` 上采样，输出 `tanh`。
  - 约定：输入图像使用 `Normalize(mean=0.5, std=0.5)`，范围约为 [-1, 1]，与 `tanh` 输出对齐。
  - 参考：`diffusion/model/vae/models/celeba.py:12`

- `models/mnist.py`
  - `MnistVAE`：用于演示“脚本复用”的 MNIST VAE。
  - 输出层用 `sigmoid`，配合 `recon_loss="bce"`。
  - 参考：`diffusion/model/vae/models/mnist.py:10`

### 模块（modules/）

- `modules/conv_blocks.py`
  - `ConvBlock`：`Conv2d + (BatchNorm) + (ReLU/LeakyReLU)`
  - `DeconvBlock`：`ConvTranspose2d + (BatchNorm) + (ReLU/LeakyReLU)`
  - 参考：`diffusion/model/vae/modules/conv_blocks.py:6`

### 数据加载（loader/）

- `loader/image_dir.py`
  - `ImageDirDataset`：递归扫描目录下图片文件（jpg/png/webp/...），逐张读取并执行 transform，返回 `Tensor`。
  - 支持 `limit` 截断，方便 smoke test。
  - 参考：`diffusion/model/vae/loader/image_dir.py:11`

- `loader/celeba.py`
  - `CelebaDataConfig` + `build_celeba_datasets`：
    - 默认从 `data_root/img_align_celeba/` 读取（存在则优先），否则直接用 `data_root` 当图片目录。
    - transform：`CenterCrop(178) -> Resize(image_size) -> ToTensor -> Normalize(0.5,0.5)`。
    - 使用 `random_split` 做 train/test 拆分（可控 `seed`）。
  - 参考：`diffusion/model/vae/loader/celeba.py:14`

- `loader/mnist.py`
  - `MnistDataConfig` + `build_mnist_datasets`：torchvision MNIST，自动下载。
  - 参考：`diffusion/model/vae/loader/mnist.py:11`

### 工具（utils/）

- `utils/device.py`
  - `get_device()`：按 `cuda -> mps -> cpu` 自动选择设备，也支持显式指定。
  - 参考：`diffusion/model/vae/utils/device.py:6`

- `utils/seed.py`
  - `seed_everything()`：设定随机种子（Python/NumPy/Torch）。
  - 参考：`diffusion/model/vae/utils/seed.py:10`

- `utils/checkpoint.py`
  - `save_checkpoint()`：保存 `model/optimizer/epoch/step` 以及额外字段。
  - `load_checkpoint()`：加载并回填到 model/optimizer。
  - checkpoint 格式：
    - `model`: `state_dict`
    - `optimizer`: `state_dict`（可选）
    - `epoch`, `step`（可选）
    - 训练参数等会被 `train.py` 放在额外字段里
  - 参考：`diffusion/model/vae/utils/checkpoint.py:9`

## 设计与运行流程

### VAE 的核心原理（对应实现）

给定输入图像 $x$，VAE 学习一个近似后验 $q_\phi(z\mid x)$（编码器），以及生成模型 $p_\theta(x\mid z)$（解码器）。

训练目标是最大化 ELBO（等价于最小化负 ELBO）：

$$
\mathcal{L}(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]}_{\text{重建项}} + \beta\,\underbrace{D_{KL}(q_\phi(z\mid x)\,\Vert\,p(z))}_{\text{KL 正则}}
$$

在代码里：

- 编码器输出 `mu, logvar`（对角高斯）：`VAEBase.encode()`
- 通过重参数技巧采样：`z = mu + eps * exp(0.5*logvar)`，见 `diffusion/model/vae/models/base.py:47`
- 解码器输出 `recon = decode(z)`
- KL 项：`-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`，见 `diffusion/model/vae/models/base.py:58`
- 重建项：
  - MSE：`F.mse_loss(reduction='none')` 后对像素求和
  - BCE：`F.binary_cross_entropy(reduction='none')` 后对像素求和
  - 见 `diffusion/model/vae/models/base.py:61`

### 训练流程（train.py）

1. 解析参数（数据集、模型、超参、输出目录等）
2. `seed_everything` + `get_device`
3. 构建数据集与 DataLoader
4. `create_model()` 创建具体模型，并设置 `beta`/`recon_loss`
5. 训练循环：
   - 前向：`recon, mu, logvar = model(x)`
   - 损失：`model.compute_loss(recon, x, mu, logvar)`
   - 反向 + `Adam` 更新（可选 AMP）
6. 每个 epoch 在测试集上跑一遍平均 loss
7. 保存 checkpoint：按 `epoch_XXXX.pt` 和 `last.pt`

## 使用方法

### 运行方式建议

建议使用模块方式运行：

```bash
python -m diffusion.model.vae.train ...
```

原因：本工程脚本使用了包内相对导入（例如 `from .loader...`），用 `-m` 最稳定。

### CelebA：训练人脸生成

数据准备：

- `--data-root` 指向 CelebA 根目录，满足任一形式：
  - `data_root/img_align_celeba/*.jpg`（推荐）
  - 或 `data_root` 直接就是图片目录

训练：

```bash
python -m diffusion.model.vae.train \
  --dataset celeba \
  --data-root /path/to/celeba \
  --image-size 64 \
  --latent-dim 256 \
  --base-channels 32 \
  --batch-size 128 \
  --epochs 20 \
  --lr 2e-4 \
  --beta 1.0
```

输出：

- checkpoint：`runs/vae/celeba_celeba_z256_s64/epoch_XXXX.pt` 与 `runs/vae/celeba_celeba_z256_s64/last.pt`

测试（会保存重建与采样网格图）：

```bash
python -m diffusion.model.vae.test \
  --dataset celeba \
  --data-root /path/to/celeba \
  --ckpt runs/vae/celeba_celeba_z256_s64/last.pt \
  --out-dir runs/vae_eval/celeba
```

推理采样：

```bash
python -m diffusion.model.vae.infer \
  --dataset celeba \
  --ckpt runs/vae/celeba_celeba_z256_s64/last.pt \
  --out runs/vae_eval/celeba/sample.png \
  --n 64
```

### MNIST：复用示例

训练：

```bash
python -m diffusion.model.vae.train \
  --dataset mnist \
  --model mnist \
  --data-root data \
  --image-size 28 \
  --latent-dim 8 \
  --batch-size 64 \
  --epochs 5
```

测试/推理与 CelebA 相同，只需要切换 `--dataset mnist` 并给定 checkpoint。

## 如何扩展到新数据集

1. 在 `loader/` 下新增 `your_dataset.py`，提供类似：
   - `YourDataConfig`
   - `build_your_datasets(cfg) -> (train_ds, test_ds)`
2. 在 `models/` 下新增 `your_dataset.py`，继承 `VAEBase` 并实现：
   - `encode(x) -> (mu, logvar)`
   - `decode(z) -> recon`
3. 在 `factory.py:create_model()` 注册新模型名称。
4. 在 `train.py/test.py` 中按 `--dataset` 分支接入新 loader（照着 `celeba/mnist` 分支添加即可）。

