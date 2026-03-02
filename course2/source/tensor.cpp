//
// Created by fss on 22-11-12.
//
// Tensor 类的实现文件
// Tensor 是深度学习推理框架中的核心数据结构，用于存储和操作多维张量数据
// 底层使用 Armadillo 库的 fcube（三维浮点数立方体）来存储数据
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace kuiper_infer {

// ==================== 构造函数 ====================
/**
 * @brief 三维张量构造函数
 * @param channels 通道数（对应 fcube 的 slices 维度）
 * @param rows 行数
 * @param cols 列数
 * @note Armadillo 的 fcube 构造参数顺序为 (rows, cols, slices)
 *       但在深度学习中通常使用 (channels, rows, cols) 的顺序
 */
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  // 创建 arma::fcube 对象，注意参数顺序是 (rows, cols, channels)
  data_ = arma::fcube(rows, cols, channels);

  // 根据 shape 的维度记录原始形状
  // 一维张量：只有列（cols）
  // 二维张量：行和列（rows, cols）
  // 三维张量：通道、行和列（channels, rows, cols）
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

/**
 * @brief 一维张量构造函数
 * @param size 张量的大小（元素个数）
 * @note 内部使用 1xsize 的二维矩阵存储，但记录为一维形状
 */
Tensor<float>::Tensor(uint32_t size) {
  // 创建 1xsize 的二维矩阵，实际上是一维向量
  data_ = arma::fcube(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

/**
 * @brief 二维张量构造函数
 * @param rows 行数
 * @param cols 列数
 * @note 内部使用单通道的 fcube 存储
 */
Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  // 创建 rowsxcols 的单通道二维矩阵
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

/**
 * @brief 通用构造函数，支持一维、二维、三维张量
 * @param shapes 形状向量，可以是 {cols}, {rows, cols}, 或 {channels, rows, cols}
 * @note 会自动补齐维度到三维，不足的维度用1填充
 */
Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  // 检查形状向量非空且维度不超过3
  CHECK(!shapes.empty() && shapes.size() <= 3);

  // 计算需要补齐的维度数，将输入形状扩展到3维
  // 例如：输入 {5} -> {1, 1, 5}，输入 {3, 4} -> {1, 3, 4}
  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  // 解析三个维度：channels, rows, cols
  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  // 创建对应大小的 fcube
  data_ = arma::fcube(rows, cols, channels);

  // 根据实际维度记录原始形状
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

// ==================== 拷贝和移动构造函数 ====================

/**
 * @brief 拷贝构造函数
 * @param tensor 源张量
 * @note 深拷贝，会复制数据
 */
Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;           // 复制数据（arma::fcube 会进行深拷贝）
    this->raw_shapes_ = tensor.raw_shapes_; // 复制形状信息
  }
}

/**
 * @brief 移动构造函数
 * @param tensor 源张量（右值引用）
 * @note 使用移动语义，避免数据复制，提高性能
 */
Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);   // 移动数据所有权
    this->raw_shapes_ = tensor.raw_shapes_;  // 复制形状信息
  }
}

/**
 * @brief 移动赋值运算符
 * @param tensor 源张量（右值引用）
 * @return 当前张量的引用
 * @note 使用移动语义，避免数据复制
 */
Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

/**
 * @brief 拷贝赋值运算符
 * @param tensor 源张量
 * @return 当前张量的引用
 * @note 深拷贝，会复制数据
 */
Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

// ==================== 维度信息获取 ====================

/**
 * @brief 获取张量的行数
 * @return 行数
 */
uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

/**
 * @brief 获取张量的列数
 * @return 列数
 */
uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

/**
 * @brief 获取张量的通道数
 * @return 通道数（对应 fcube 的 slices 维度）
 */
uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;  // 在 fcube 中，通道对应 slices 维度
}

/**
 * @brief 获取张量的总元素个数
 * @return 总元素数 = channels * rows * cols
 */
uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

// ==================== 数据设置和访问 ====================

/**
 * @brief 设置张量数据
 * @param data 新的 arma::fcube 数据
 * @note 新数据的形状必须与当前张量形状一致
 */
void Tensor<float>::set_data(const arma::fcube& data) {
  // 检查维度是否匹配
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

/**
 * @brief 判断张量是否为空
 * @return true 如果张量为空，否则 false
 */
bool Tensor<float>::empty() const { return this->data_.empty(); }

/**
 * @brief 按线性索引获取元素值（只读版本）
 * @param offset 线性偏移量（0 到 size()-1）
 * @return 指定位置的元素值
 * @note 线性索引按照列优先顺序访问（Armadillo 默认存储顺序）
 */
float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);  // at() 方法会进行边界检查
}

/**
 * @brief 按线性索引获取元素引用（可修改版本）
 * @param offset 线性偏移量（0 到 size()-1）
 * @return 指定位置的元素引用，可以用于修改元素值
 */
float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

/**
 * @brief 获取张量的形状
 * @return 包含三个维度的向量 {channels, rows, cols}
 */
std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

/**
 * @brief 获取底层数据的引用（可修改版本）
 * @return arma::fcube 的引用
 */
arma::fcube& Tensor<float>::data() { return this->data_; }

/**
 * @brief 获取底层数据的常量引用（只读版本）
 * @return arma::fcube 的常量引用
 */
const arma::fcube& Tensor<float>::data() const { return this->data_; }

/**
 * @brief 获取指定通道的矩阵切片（可修改版本）
 * @param channel 通道索引
 * @return 该通道对应的二维矩阵引用
 * @note 可用于直接操作某个通道的数据
 */
arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

/**
 * @brief 获取指定通道的矩阵切片（只读版本）
 * @param channel 通道索引
 * @return 该通道对应的二维矩阵常量引用
 */
const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

/**
 * @brief 获取指定位置的元素值（只读版本）
 * @param channel 通道索引
 * @param row 行索引
 * @param col 列索引
 * @return 指定位置的元素值
 */
float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

/**
 * @brief 获取指定位置的元素引用（可修改版本）
 * @param channel 通道索引
 * @param row 行索引
 * @param col 列索引
 * @return 指定位置的元素引用，可用于修改元素值
 */
float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

// ==================== 张量操作方法 ====================

/**
 * @brief 对张量进行填充（Padding）
 * @param pads 填充大小向量 {上, 下, 左, 右}
 * @param padding_value 填充值，默认为0
 * @note 常用于卷积操作前的边缘填充
 *
 * 示例：对于 3x3 的矩阵，pads = {1,1,1,1} 填充后变成 5x5
 *
 * 原始矩阵:          填充后:
 *     [a b c]        [0 0 0 0 0]
 *     [d e f]   ->   [0 a b c 0]
 *     [g h i]        [0 d e f 0]
 *                    [0 g h i 0]
 *                    [0 0 0 0 0]
 */
void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);

  // 解析四个方向的填充大小
  uint32_t pad_rows1 = pads.at(0);  // 上边填充行数
  uint32_t pad_rows2 = pads.at(1);  // 下边填充行数
  uint32_t pad_cols1 = pads.at(2);  // 左边填充列数
  uint32_t pad_cols2 = pads.at(3);  // 右边填充列数

  // 如果没有填充，直接返回
  if (pad_rows1 == 0 && pad_rows2 == 0 && pad_cols1 == 0 && pad_cols2 == 0) {
    return;
  }

  // 计算填充后的新尺寸
  const uint32_t new_rows = this->rows() + pad_rows1 + pad_rows2;
  const uint32_t new_cols = this->cols() + pad_cols1 + pad_cols2;
  const uint32_t channels = this->channels();

  // 创建新的 fcube 并用填充值初始化
  arma::fcube new_data(new_rows, new_cols, channels);
  new_data.fill(padding_value);

  // 将原始数据复制到新 fcube 的中间位置
  for (uint32_t c = 0; c < channels; ++c) {
    // 获取新矩阵中对应通道的切片
    arma::fmat& new_slice = new_data.slice(c);
    // 将原始数据复制到正确的位置
    // 起始位置为 (pad_rows1, pad_cols1)
    new_slice.submat(pad_rows1, pad_cols1,
                     pad_rows1 + this->rows() - 1,
                     pad_cols1 + this->cols() - 1) = this->data_.slice(c);
  }

  // 更新数据
  this->data_ = std::move(new_data);

  // 更新形状信息
  this->raw_shapes_ = {channels, new_rows, new_cols};
}

/**
 * @brief 用指定值填充整个张量
 * @param value 填充值
 */
void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

/**
 * @brief 用向量中的值填充张量
 * @param values 值向量，大小必须与张量元素总数一致
 * @param row_major 是否按行主序填充，默认为 true
 *                  - true: 按行主序（C风格），每行内元素连续
 *                  - false: 按列主序（Fortran风格），每列内元素连续
 * @note Armadillo 默认使用列主序存储，但深度学习框架通常使用行主序
 */
void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  if (row_major) {
    // 行主序填充：需要转置处理
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;  // 每个通道的元素数
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);
      // 按行主序构造矩阵，然后转置以适应 Armadillo 的列主序存储
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  } else {
    // 列主序填充：直接复制到内存
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

/**
 * @brief 打印张量内容到日志
 * @note 按通道分别打印每个通道的矩阵内容
 */
void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

/**
 * @brief 将张量展平为一维
 * @param row_major 是否按行主序展平
 *                  - true: 按行主序展平（C风格）
 *                  - false: 按列主序展平（Fortran风格）
 * @note 展平后张量形状变为 {total_size}，即一维向量
 */
void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());

  // 1. 获取所有元素值，根据 row_major 决定顺序
  std::vector<float> values = this->values(row_major);

  // 2. Reshape 为一维形状 {total_size}
  const uint32_t total_size = this->size();
  this->data_.reshape(1, total_size, 1);
  this->raw_shapes_ = {total_size};

  // 3. 按照原来的顺序填充数据
  this->Fill(values, row_major);
}

/**
 * @brief 用随机数填充张量
 * @note 使用标准正态分布 N(0,1) 生成随机数
 */
void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();  // Armadillo 的 randn() 生成标准正态分布随机数
}

/**
 * @brief 将张量所有元素设置为1
 */
void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

/**
 * @brief 对张量中的每个元素应用变换函数
 * @param filter 变换函数，接受一个 float 值，返回变换后的 float 值
 * @note 常用于激活函数、归一化等操作
 *
 * 示例：
 *   tensor.Transform([](float x) { return x > 0 ? x : 0; });  // ReLU
 */
void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

/**
 * @brief 获取张量的原始形状（不考虑内部存储的补齐）
 * @return 原始形状向量的常量引用
 *         - 一维：{cols}
 *         - 二维：{rows, cols}
 *         - 三维：{channels, rows, cols}
 * @note 与 shapes() 不同，raw_shapes() 返回的是构造时的原始维度
 */
const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

/**
 * @brief 重新调整张量的形状
 * @param shapes 新的形状向量
 * @param row_major 是否保持行主序的数据排列
 *                  - true: 先提取行主序数据，reshape 后再按行主序填充
 *                  - false: 直接 reshape，数据顺序可能改变
 * @note 新形状的元素总数必须与原形状相同
 *
 * 示例：
 *   tensor.Reshape({3, 4});  // 将张量 reshape 为 3x4 的二维张量
 */
void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());

  // 检查新旧形状的元素总数是否一致
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  // 如果需要保持行主序，先保存原始数据
  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }

  // 根据新形状的维度数进行 reshape
  if (shapes.size() == 3) {
    // 三维：{channels, rows, cols}
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    // 二维：{rows, cols}
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    // 一维：{size}
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shapes_ = {shapes.at(0)};
  }

  // 如果需要保持行主序，重新填充数据
  if (row_major) {
    this->Fill(values, true);
  }
}

// ==================== 原始指针访问 ====================

/**
 * @brief 获取底层数据的原始指针
 * @return 指向第一个元素的指针
 * @note 可用于与 C API 或其他库进行交互
 */
float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();  // 返回内存首地址
}

/**
 * @brief 获取指定偏移位置的原始指针
 * @param offset 偏移量（元素个数）
 * @return 指向偏移位置的指针
 */
float* Tensor<float>::raw_ptr(uint32_t offset) {
  const uint32_t size = this->size();
  CHECK(!this->data_.empty());
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;  // 返回偏移后的地址
}

/**
 * @brief 获取张量中的所有元素值
 * @param row_major 是否按行主序返回
 *                  - true: 按行主序（C风格）返回
 *                  - false: 按列主序（Fortran风格）返回
 * @return 包含所有元素的向量
 */
std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<float> values(this->data_.size());

  if (!row_major) {
    // 列主序：直接复制内存数据
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    // 行主序：需要按通道逐个转置后复制
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      // 获取该通道矩阵的转置（行主序）
      const arma::fmat& channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

/**
 * @brief 获取指定通道的原始数据指针
 * @param index 通道索引
 * @return 指向该通道第一个元素的指针
 * @note 可用于直接访问某个通道的连续数据
 */
float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  // 计算该通道在内存中的偏移量
  uint32_t offset = index * this->rows() * this->cols();
  CHECK_LE(offset, this->size());
  float* mem_ptr = this->raw_ptr() + offset;
  return mem_ptr;
}

}  // namespace kuiper_infer
